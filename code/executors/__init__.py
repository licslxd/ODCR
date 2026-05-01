"""
ODCR 执行体模块层：Step3/4/5 核心实现与 torchrun 入口。

- ``step3_train_core`` / ``step4_engine`` / ``step5_engine``：业务与训练主逻辑（重型依赖）。
- ``step3_entry`` / ``step4_entry`` / ``step5_entry``：argparse + 调用 core（可被薄壳脚本引用）。
- 对外用户入口仍为 ``code/odcr.py``；``code/`` 下若干 ``*.py`` 薄壳仅用于 torchrun 加载历史文件名。
"""
