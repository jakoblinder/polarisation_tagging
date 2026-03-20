from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


@dataclass
class Parameter:
    name: str
    value: Any
    fit: Any = False

    def __post_init__(self) -> None:
        self.fit = self._validate_fit(self.fit)

    @staticmethod
    def _validate_fit(fit: Any) -> Any:
        # fit=False means no fit; any non-bool payload encodes fit values/ranges.
        if isinstance(fit, bool):
            if fit:
                raise ValueError("fit=True is ambiguous. Use fit=False or provide fit values/range.")
            return False
        return fit

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Parameter":
        return cls(
            name  = data["name"],
            value = data.get("value"),
            fit   = data.get("fit", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["value"] = Settings._encode_yaml_value(data["value"])
        data["fit"] = Settings._encode_yaml_value(data["fit"])
        return data


class Settings:
    def __init__(self,argparse=None, **kwargs):
        # Keep parameters in a dedicated mapping to simplify robust I/O.
        super().__setattr__("_parameters", {})

        if argparse:
            for key, value in vars(argparse).items():
                self._set_parameter(key, value)

        for key, value in kwargs.items():
            if isinstance(value, Parameter):
                self._set_parameter(key, value.value, fit=value.fit)
            else:
                self._set_parameter(key, value)

    def _set_parameter(self, key: str, value: Any, fit: Any = False) -> None:
        self._parameters[key] = Parameter(name=key, value=value, fit=fit)

    def set(self,
            key: str,
            value: Any,
            fit: Any = False,
            overwrite: bool = False) -> None:
        if not overwrite and key in self._parameters:
            raise KeyError(f"Setting '{key}' already exists.")
        self._set_parameter(key, value, fit=fit)

    def update_fit(self, key: str, fit: Any) -> None:
        if key not in self._parameters:
            raise KeyError(f"Unknown setting '{key}'.")
        self._parameters[key].fit = Parameter._validate_fit(fit)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            super().__setattr__(key, value)
            return

        if isinstance(value, Parameter):
            self._set_parameter(key, value.value, fit=value.fit)
        else:
            self._set_parameter(key, value)

    def __getattr__(self, key: str) -> Parameter:
        try:
            return self._parameters[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __getitem__(self, key: str) -> Parameter:
        return self._parameters[key]

    def keys(self) -> Iterable[str]:
        return self._parameters.keys()

    def items(self):
        return self._parameters.items()

    def clone(self) -> "Settings":
        return Settings.from_dict(self.to_dict())

    def with_overrides(self,
                       overrides: Dict[str, Any],
                       overwrite: bool = True) -> "Settings":
        clone = self.clone()
        for key, value in overrides.items():
            if isinstance(value, Parameter):
                clone.set(key, value.value, fit=value.fit, overwrite=overwrite)
            else:
                clone.set(key, value, overwrite=overwrite)
        return clone

    def log_to_logger(self,
                      logger,
                      header: str = "Run settings:",
                      include_fit: bool = False) -> None:
        logger.info(header)
        for key, parameter in self._parameters.items():
            if include_fit:
                logger.info(f"  {key}: value={parameter.value}, fit={parameter.fit}")
            else:
                logger.info(f"  {key}: {parameter.value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": {
                name: parameter.to_dict()
                for name, parameter in self._parameters.items()
            }
        }

    @staticmethod
    def _encode_yaml_value(value: Any) -> Any:
        if isinstance(value, Path):
            return {"__type__": "path", "value": str(value)}
        if isinstance(value, tuple):
            return {
                "__type__": "tuple",
                "value": [Settings._encode_yaml_value(item) for item in value],
            }
        if isinstance(value, list):
            return [Settings._encode_yaml_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: Settings._encode_yaml_value(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _decode_yaml_value(value: Any) -> Any:
        if isinstance(value, dict):
            value_type = value.get("__type__")
            if value_type == "path":
                return Path(value["value"])
            if value_type == "tuple":
                return tuple(Settings._decode_yaml_value(item) for item in value["value"])
            return {
                key: Settings._decode_yaml_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [Settings._decode_yaml_value(item) for item in value]
        return value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        instance = cls()
        parameters = data.get("parameters", {})

        for key, parameter_data in parameters.items():
            if isinstance(parameter_data, dict):
                if "name" not in parameter_data:
                    parameter_data = {"name": key, **parameter_data}
                parameter_data = {
                    "name": parameter_data["name"],
                    "value": cls._decode_yaml_value(parameter_data.get("value")),
                    "fit": cls._decode_yaml_value(parameter_data.get("fit", False)),
                }
                parameter = Parameter.from_dict(parameter_data)
                instance._set_parameter(key, parameter.value, fit=parameter.fit)
            else:
                # Backward-compatible shape: parameter as scalar value.
                instance._set_parameter(key, cls._decode_yaml_value(parameter_data))

        return instance

    def dump_yaml(self, file_path: str | Path) -> None:
        path = Path(file_path)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.to_dict(),
                handle,
                sort_keys=True,
                default_flow_style=False,
                allow_unicode=False,
            )

    @classmethod
    def load_yaml(cls, file_path: str | Path) -> "Settings":
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML settings content must be a mapping at top level.")
        return cls.from_dict(data)

