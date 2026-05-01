"""Step3 执行体 API（入口实现见 ``step3_entry``，核心见 ``step3_train_core``）。"""

from executors.step3_entry import print_step3_root_help, run_step3_cli

run_step3_main = run_step3_cli

__all__ = ["print_step3_root_help", "run_step3_cli", "run_step3_main"]
