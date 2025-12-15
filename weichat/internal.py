#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
断网自愈（SSH 版）：
- 每隔 CHECK_INTERVAL_SEC 秒检测外网
- 连续失败 FAILURE_THRESHOLD 次，走 SSH 执行路由器重启命令
- 设置冷却时间，避免频繁重启
"""

import socket, time, urllib.request
from contextlib import closing
from typing import Optional

# ========== 用户配置区 ==========
ROUTER_SSH_HOST = "192.168.1.1"
ROUTER_SSH_PORT = 22
ROUTER_SSH_USER = "admin"
# 任选其一：A) 用密码  B) 用密钥（推荐）
ROUTER_SSH_PASSWORD: Optional[str] = 'myf1996521'     # 如需密码登录，填入字符串；否则 None
ROUTER_SSH_KEYFILE: Optional[str] = "/home/pi/.ssh/id_ed25519"  # 密钥路径；若用密码，把它设为 None

ROUTER_SSH_REBOOT_CMD = "reboot"  # OpenWrt/大多数 Linux 路由器可用

CHECK_INTERVAL_SEC = 60           # 检测间隔
FAILURE_THRESHOLD = 3             # 连续失败 N 次触发重启
REBOOT_COOLDOWN_SEC = 15 * 60     # 重启后冷却时间

PING_TARGETS = [("8.8.8.8", 53), ("1.1.1.1", 53)]
HTTP_TARGETS = ["http://www.google.com", "http://example.com"]  # 国内可换 http://www.baidu.com
SOCKET_TIMEOUT = 3
HTTP_TIMEOUT = 5
# ==============================

def check_tcp(host: str, port: int, timeout: float) -> bool:
    try:
        with closing(socket.create_connection((host, port), timeout=timeout)):
            return True
    except Exception:
        return False

def check_http(url: str, timeout: float) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NetHeal/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.getcode() < 400
    except Exception:
        return False

def internet_is_up() -> bool:
    if not any(check_tcp(h, p, SOCKET_TIMEOUT) for (h, p) in PING_TARGETS):
        return False
    if not any(check_http(u, HTTP_TIMEOUT) for u in HTTP_TARGETS):
        return False
    return True

def reboot_via_ssh(host: str, port: int, user: str,
                   password: Optional[str], keyfile: Optional[str],
                   cmd: str) -> None:
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if keyfile:
        client.connect(host, port=port, username=user, key_filename=keyfile, timeout=10)
    elif password:
        client.connect(host, port=port, username=user, password=password, timeout=10)
    else:
        # 走默认密钥
        client.connect(host, port=port, username=user, timeout=10)
    try:
        _, stdout, stderr = client.exec_command(cmd)
        _ = stdout.read(); _ = stderr.read()
    finally:
        client.close()

def main():
    consecutive_fail = 0
    last_reboot_ts = 0.0

    while True:
        if internet_is_up():
            consecutive_fail = 0
            print("good")
        else:
            consecutive_fail += 1
            print(f"[WARN] Internet check failed {consecutive_fail}/{FAILURE_THRESHOLD}")

        should_reboot = (
            consecutive_fail >= FAILURE_THRESHOLD and
            (time.time() - last_reboot_ts >= REBOOT_COOLDOWN_SEC)
        )

        if True:
            print("[ACTION] Rebooting router via SSH ...")
            try:
                reboot_via_ssh(
                    ROUTER_SSH_HOST, ROUTER_SSH_PORT, ROUTER_SSH_USER,
                    ROUTER_SSH_PASSWORD, ROUTER_SSH_KEYFILE, ROUTER_SSH_REBOOT_CMD
                )
                last_reboot_ts = time.time()
                consecutive_fail = 0
                print("[DONE] Reboot command sent. Cooling down ...")
            except Exception as e:
                print(f"[ERROR] SSH reboot failed: {e}")

        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
