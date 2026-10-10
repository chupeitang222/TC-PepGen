"""CSV export regressions without model weights or ML dependencies."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def batch_inference():
    # Bypass eager package exports and stub only ML imports. The actual Config
    # and CSV column resolvers are still used by the batch entry point.
    modules = {}
    for name, relative_path in [
        ("tc_pepgen", "tc_pepgen"),
        ("tc_pepgen.inference", "tc_pepgen/inference"),
        ("tc_pepgen.utils", "tc_pepgen/utils"),
        ("tc_pepgen.models", "tc_pepgen/models"),
    ]:
        package = ModuleType(name)
        package.__path__ = [str(ROOT / "src" / relative_path)]
        modules[name] = package

    torch = ModuleType("torch")
    torch.LongTensor = Mock()
    torch.FloatTensor = Mock()
    modules["torch"] = torch
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = Mock()
    transformers.EncoderDecoderModel = Mock()
    transformers.LogitsProcessor = type("LogitsProcessor", (), {})
    modules["transformers"] = transformers
    model = ModuleType("tc_pepgen.models.hf_encoder_decoder_model")
    model.EsmWrapperConfig = Mock()
    model.EsmWrapperForEncoderDecoder = Mock()
    modules[model.__name__] = model

    spec = importlib.util.spec_from_file_location(
        "batch_inference_under_test", ROOT / "scripts" / "batch_inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path.copy()):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["topk", "target_length_mode"])
@pytest.mark.parametrize("path_kind", ["relative", "absolute", "filename", "existing"])
def test_saves_csv(batch_inference, monkeypatch, tmp_path, mode, path_kind):
    monkeypatch.chdir(tmp_path)
    # Config exposes DATA_INPUT_COLUMN, while the resolver expects INPUT_COLUMN.
    # Supply that alias in the test to isolate this CSV export regression from
    # the existing configuration mismatch without changing production scope.
    config = batch_inference.Config()
    config.INPUT_COLUMN = config.DATA_INPUT_COLUMN
    monkeypatch.setattr(batch_inference, "Config", Mock(return_value=config))
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "protein_sequence": [" acde ", " fghi "],
            "target_length": [3, None],
            "binder_sequence": [None, " kl-mn "],
        }
    ).to_csv(input_csv, index=False)

    output_csv = Path("outputs/nested/batch_results.csv")
    if path_kind == "absolute":
        output_csv = tmp_path / "absolute" / "nested" / "batch_results.csv"
    elif path_kind == "filename":
        output_csv = Path("batch_results.csv")
    elif path_kind == "existing":
        output_csv.parent.mkdir(parents=True)
        output_csv.write_text("old contents\n", encoding="utf-8")

    if path_kind in ("relative", "absolute"):
        assert not output_csv.parent.exists()

    inference = Mock()
    inference.predict.side_effect = [["KLM"], ["NPQR"]]
    inference.generate_with_target_length.side_effect = ["KLM", "NPQR"]
    factory = Mock(return_value=inference)
    monkeypatch.setattr(batch_inference, "ProteinInference", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_inference.py", str(input_csv),
            "--output_csv", str(output_csv),
            "--model_path", "unused-model", "--mode", mode, "--top_k", "5",
        ],
    )

    batch_inference.main()

    factory.assert_called_once_with("unused-model")
    inference.load_model.assert_called_once_with()
    if mode == "topk":
        assert inference.predict.call_args_list == [
            call("ACDE", num_sequences=1, top_k=5),
            call("FGHI", num_sequences=1, top_k=5),
        ]
        inference.generate_with_target_length.assert_not_called()
    else:
        assert inference.generate_with_target_length.call_args_list == [
            call("ACDE", 3, top_k=5), call("FGHI", 4, top_k=5),
        ]
        inference.predict.assert_not_called()

    assert output_csv.is_file()
    pd.testing.assert_frame_equal(
        pd.read_csv(output_csv),
        pd.DataFrame(
            {"input_sequence": ["ACDE", "FGHI"], "predicted_sequence": ["KLM", "NPQR"]}
        ),
    )
