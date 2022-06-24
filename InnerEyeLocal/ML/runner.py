from pathlib import Path
import os
from InnerEye.ML import runner


def main() -> None:
    current = os.path.dirname(os.path.realpath(__file__))
    project_root = Path(os.path.realpath(os.path.join(current, "..", "..")))
    runner.run(project_root=project_root,
               yaml_config_file=project_root / "InnerEyeLocal/settings.yml",
               post_cross_validation_hook=None)


if __name__ == '__main__':
    main() 