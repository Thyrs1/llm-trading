"""
本模板自 Nautilus Trader 重构后不再推荐使用。

请复制 `config.toml.example` 为 `config.toml` 并填写密钥，或通过环境变量覆盖：

```
cp config.toml.example config.toml
```
"""

from nautilus_bot.config import BotSettings, load_settings


def sample_settings() -> BotSettings:
    """输出当前默认配置，便于了解字段结构。"""

    return load_settings()


if __name__ == "__main__":
    settings = sample_settings()
    print("默认配置预览：")
    print(settings)
