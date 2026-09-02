# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from megatron.core import fp8_utils
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_quantized_param_init_memory_context_is_nested_and_scoped():
    events = []

    @contextmanager
    def memory_context():
        events.append("memory-enter")
        yield
        events.append("memory-exit")

    @contextmanager
    def te_context():
        events.append("te-enter")
        yield
        events.append("te-exit")

    with fp8_utils.quantized_param_init_memory_context(memory_context):
        with fp8_utils._with_quantized_param_init_memory_context(te_context()):
            events.append("body")

    assert events == ["memory-enter", "te-enter", "body", "te-exit", "memory-exit"]

    events.clear()
    with fp8_utils._with_quantized_param_init_memory_context(nullcontext()):
        events.append("unscoped")
    assert events == ["unscoped"]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"fp8_backward_override": "high_precision"}, "together with fp8 mode"),
        (
            {
                "fp8": "hybrid",
                "fp8_recipe": "delayed",
                "fp8_backward_override": "high_precision",
            },
            "mxfp8.*only",
        ),
        (
            {
                "fp8": "hybrid",
                "fp8_recipe": "blockwise",
                "fp8_backward_override": "dequantized",
            },
            "mxfp8.*only",
        ),
        (
            {
                "fp8": "hybrid",
                "fp8_recipe": "mxfp8",
                "fp8_param": True,
                "omit_columnwise_primary_weight_storage": True,
            },
            "requires fp8_backward_override",
        ),
    ],
)
def test_transformer_config_rejects_invalid_fp8_backward_override_combinations(kwargs, error):
    with pytest.raises(ValueError, match=error):
        TransformerConfig(num_layers=1, num_attention_heads=1, **kwargs)


@pytest.mark.skipif(not fp8_utils.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.parametrize(
    ("backward_override", "omit_columnwise"),
    [
        (None, False),
        ("high_precision", False),
        ("high_precision", True),
        ("dequantized", True),
    ],
)
def test_fp8_init_uses_explicit_columnwise_storage_contract(backward_override, omit_columnwise):
    config = TransformerConfig(
        num_layers=1,
        num_attention_heads=1,
        fp8="hybrid",
        fp8_recipe="mxfp8",
        fp8_param=True,
        fp8_backward_override=backward_override,
        omit_columnwise_primary_weight_storage=omit_columnwise,
    )
    constructor_args = {}
    init_args = {}

    def recipe_constructor(*, fp8_format, fp8_dpa=False, backward_override=None):
        constructor_args["backward_override"] = backward_override
        return SimpleNamespace(backward_override=backward_override, mxfp8=lambda: True)

    def model_init(
        *,
        enabled=True,
        recipe=None,
        preserve_high_precision_init_val=False,
        omit_columnwise_primary_weight_storage=False,
    ):
        init_args["recipe"] = recipe
        init_args["omit_columnwise"] = omit_columnwise_primary_weight_storage
        return nullcontext()

    with (
        patch.object(fp8_utils, "is_te_min_version", return_value=True),
        patch.object(
            fp8_utils.transformer_engine.common.recipe,
            "MXFP8BlockScaling",
            recipe_constructor,
        ),
        patch.object(fp8_utils.parallel_state, "model_parallel_is_initialized", return_value=False),
        patch.object(fp8_utils.transformer_engine.pytorch, "quantized_model_init", model_init),
    ):
        with fp8_utils.get_fp8_context(config, is_init=True):
            pass

    assert constructor_args["backward_override"] == backward_override
    assert init_args["recipe"].backward_override == backward_override
    assert init_args["omit_columnwise"] is omit_columnwise


class MockTELinear(nn.Module):
    """Mock TE Linear module for testing."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x @ self.weight.t()


class TestFP8Padding:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        # Clear the wrapped modules set before each test
        fp8_utils._fp8_inference_wrapped_modules.clear()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        fp8_utils._fp8_inference_wrapped_modules.clear()

    def test_prepare_model_for_fp8_inference_basic(self):
        """Test prepare_model_for_fp8_inference wraps TE modules."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.te_layer = MockTELinear(128, 128)
                self.regular_layer = nn.Linear(128, 128)

        with (
            patch.object(fp8_utils, 'HAVE_TE', True),
            patch.object(fp8_utils, 'Fp8Padding'),
            patch.object(fp8_utils, 'Fp8Unpadding'),
            patch.object(fp8_utils, 'TE_LINEAR_TYPES', (MockTELinear,)),
        ):

            model = SimpleModel()
            original_te_forward = model.te_layer.forward
            original_regular_forward = model.regular_layer.forward

            # Prepare model
            prepared_model = fp8_utils.prepare_model_for_fp8_inference(model)

            # Check same model returned
            assert prepared_model is model

            # Check TE layer was wrapped
            assert model.te_layer.forward != original_te_forward
            assert model.te_layer in fp8_utils._fp8_inference_wrapped_modules

            # Check regular layer was not wrapped
            assert model.regular_layer.forward == original_regular_forward

    def test_padding_mechanism_works(self):
        """Test that the padding mechanism actually pads and unpads correctly."""

        with (
            patch.object(fp8_utils, 'HAVE_TE', True),
            patch.object(fp8_utils, 'Fp8Padding') as mock_pad_class,
            patch.object(fp8_utils, 'Fp8Unpadding') as mock_unpad_class,
        ):

            # Setup padding mock to pad from 6 to 16
            mock_pad_instance = Mock()
            mock_pad_instance.return_value = (torch.zeros(16, 8192), [16])
            mock_pad_class.return_value = mock_pad_instance

            # Setup unpadding mock to unpad from 16 to 6
            mock_unpad_instance = Mock()
            mock_unpad_instance.return_value = torch.zeros(6, 8192)
            mock_unpad_class.return_value = mock_unpad_instance

            # Create module and get access to padded_forward directly
            module = MockTELinear(4096, 4096)
            module.cuda()

            # Store original forward to track what it receives
            original_forward_input = None

            def track_forward(x):
                nonlocal original_forward_input
                original_forward_input = x
                return torch.randn(x.shape[0], x.shape[1], 4096).cuda()

            module.forward = track_forward

            # Manually create the wrapped forward function
            fp8_utils._wrap_te_linear_for_padding(module)
            padded_forward = module.forward

            # Mock FP8GlobalStateManager.is_fp8_enabled to return True
            with patch(
                'transformer_engine.pytorch.fp8.FP8GlobalStateManager.is_fp8_enabled',
                return_value=True,
            ):
                # Create input: (seq_len=6, batch=2, hidden=4096)
                input_tensor = torch.randn(6, 2, 4096).cuda()

                # Call padded_forward directly
                output = padded_forward(input_tensor)

            # Verify padding was called with correct reshaped input
            mock_pad_instance.assert_called_once()
            call_args = mock_pad_instance.call_args[0]
            assert call_args[0].shape == (6, 8192)  # Reshaped to 2D
            assert call_args[1] == [6]  # Split info

            # Verify the original forward received padded input with correct shape
            assert original_forward_input.shape == (16, 2, 4096)  # Padded to 16

            # Verify unpadding was called
            mock_unpad_instance.assert_called_once()
            unpad_args = mock_unpad_instance.call_args[0]
            assert unpad_args[0].shape == (16, 8192)  # Padded 2D tensor
            assert unpad_args[1] == [6]  # Original split

            # Verify output has original shape
            assert output.shape == (6, 2, 4096)  # Back to original seq_len
