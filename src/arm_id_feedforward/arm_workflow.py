#!/usr/bin/env python3
"""Interactive plan -> preview -> capture -> analysis workflow."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

if __package__:
    from . import arm_collect, arm_host
    from .arm_analysis import analyze_capture
else:
    import arm_collect
    import arm_host
    from arm_analysis import analyze_capture


STAGES = ("preparation", "dynamic", "validation", "identification", "static")
NAMES = {"preparation": "激励轨迹生成", "static": "静态采集（可选）",
         "dynamic": "训练动态采集", "validation": "独立验证采集", "identification": "完整动态辨识"}


class Cancelled(Exception):
    pass


def confirm(message: str, *, motion: bool = False) -> None:
    prompt = " [输入 执行 开始，回车取消]：" if motion else " [y/回车取消]："
    answer = input(message + prompt).strip().lower()
    if answer not in (("执行",) if motion else ("y", "yes")):
        raise Cancelled()


class ProgressLog:
    """Keep detailed tool output on disk and emit sparse progress messages."""

    def __init__(self, stream, terminal):
        self.stream = stream
        self.terminal = terminal
        self.last_progress = time.monotonic()

    def write(self, text):
        result = self.stream.write(text)
        self.stream.flush()
        now = time.monotonic()
        if text.startswith("警告："):
            print(text.rstrip(), file=self.terminal, flush=True)
            self.last_progress = now
        if now - self.last_progress >= 10:
            print("  正在处理，详细日志持续保存中…", file=self.terminal, flush=True)
            self.last_progress = now
        return result

    def flush(self):
        self.stream.flush()


def run_tool(module, arguments: list[str], log: Path) -> None:
    args = module.build_parser().parse_args(arguments)
    with log.open("a", encoding="utf-8") as stream:
        sink = ProgressLog(stream, sys.stdout)
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            print(f"\n{datetime.now().isoformat()} {module.__name__}: {arguments}")
            try:
                code = args.handler(args)
                if code == 130:
                    raise KeyboardInterrupt()
                if code:
                    raise RuntimeError(f"操作未完成，退出码 {code}")
            except BaseException:
                traceback.print_exc(file=stream)
                raise


def write_json(path: Path, document: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def choose_port(port: str | None) -> str:
    if port:
        return port
    ports = list(arm_host.list_ports.comports())
    if len(ports) == 1:
        print(f"检测到串口：{ports[0].device}")
        confirm("使用该串口？")
        return ports[0].device
    for index, entry in enumerate(ports, start=1):
        print(f"  {index}. {entry.device} — {entry.description}")
    answer = input("选择串口序号，或直接输入串口路径：").strip()
    if answer.isdigit() and 1 <= int(answer) <= len(ports):
        return ports[int(answer) - 1].device
    if not answer:
        raise Cancelled()
    return answer


class Workflow:
    def __init__(self, output: Path, port: str | None):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.port = port
        self.manifest = self.output / "session.json"
        if self.manifest.exists():
            self.state = json.loads(self.manifest.read_text(encoding="utf-8"))
            if self.state.get("kind") != "arm_identification_workflow" or self.state.get("version") != 1:
                raise ValueError("该目录不是受支持的工作流会话")
        else:
            self.state = {
                "kind": "arm_identification_workflow", "version": 1,
                "attempts": {}, "completed": [],
            }
        # Preserve legacy alignment/calibration records, but never apply them.
        self.state["torque_source"] = "raw_feedback"
        self.save()

    def save(self):
        write_json(self.manifest, self.state)

    def attempt(self, stage: str) -> tuple[Path, dict[str, object]]:
        previous = self.state["attempts"].get(stage)
        if previous and previous["phase"] in ("generated", "captured"):
            updated_plan = (
                stage != "preparation" and previous["phase"] == "generated"
                and previous.get("preparation") != self.state.get("preparation")
            )
            if updated_plan:
                print("第0步计划已更新，将创建新的执行计划，旧文件保留。")
            else:
                print("发现尚未完成的步骤，将继续已有轨迹或采集文件。")
                return self.output / previous["directory"], previous
        for index in range(1, 10000):
            directory = self.output / f"{stage}_{index:03d}"
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            record = {"directory": str(directory.relative_to(self.output)), "phase": "new"}
            self.state["attempts"][stage] = record
            self.save()
            return directory, record
        raise RuntimeError("本会话运行次数过多，请使用新目录")

    def run_stage(self, stage: str) -> bool:
        if stage not in STAGES:
            raise ValueError("未知工作流阶段")
        if stage == "preparation":
            return self.prepare()
        if stage == "identification":
            return self.identify()
        dynamic_stage = stage in ("dynamic", "validation")
        print(f"\n{NAMES[stage]}：生成轨迹 → 预览 → 实机采集 → 数据分析")
        directory, record = self.attempt(stage)
        plan = directory / "plan.csv"
        capture = directory / "capture.csv"
        log = directory / "details.log"
        print(f"本次输出：{directory}")
        if record["phase"] == "new":
            source = self.prepared_plan(stage)
            confirm("从第0步筛选结果生成本阶段执行计划？")
            shutil.copyfile(source, plan)
            shutil.copyfile(source.with_suffix(".json"), plan.with_suffix(".json"))
            record["preparation"] = self.state["preparation"]
            record["plan_sha256"] = arm_collect._sha256(plan)
            record["phase"] = "generated"
            self.save()
            print("轨迹已生成。")

        if record["phase"] == "generated":
            if not record.get("preparation") or arm_collect._sha256(plan) != record.get("plan_sha256"):
                record["phase"] = "rejected"
                self.save()
                raise ValueError("计划未关联第0步或已被修改，请重新生成执行计划")
            self.prepared_plan(stage, record["preparation"])
            confirm("打开运动预览？将读取当前姿态；关闭预览窗口后继续")
            self.port = choose_port(self.port)
            # Import the viewer only when preview is requested.
            if __package__:
                from . import arm_sim
            else:
                import arm_sim
            print("正在打开 2.5 倍速预览。实机按原轨迹时长执行；关闭窗口后还需确认。")
            run_tool(arm_sim, [
                "playback", str(plan), "--port", self.port,
                "--speed", "2.5",
                "--rate", "200", "--transition-duration", "24" if dynamic_stage else "3",
                "--hold-duration", "5" if stage == "static" else "3",
            ], log)
            print("预览已结束。实机执行会重新读取当前姿态。")
            confirm("确认预览可接受、现场已准备好，开始实机运动和采集？", motion=True)
            if dynamic_stage:
                collection = ["collect-dynamic", "--trajectory", str(plan)]
            else:
                collection = [
                    "collect-poses", "--stage", stage, "--poses", str(plan),
                    "--capture-seconds", "5" if stage == "static" else "3",
                ]
            record["torque_source"] = "raw_feedback"
            print("实机采集中。Ctrl+C 可中止；文件将保留。")
            record["phase"] = "collecting"
            self.save()
            run_tool(arm_collect, [
                *collection, "--port", self.port, "--output", str(capture),
            ], log)
            record["phase"] = "captured"
            self.save()
            print("采集已完成，原始记录和元数据已保存。")

        confirm("开始数据分析？详细数据只保存到文件")
        report = analyze_capture(capture, "dynamic" if dynamic_stage else stage, plan)
        report_path = directory / "quality_report.json"
        write_json(report_path, report)
        if not report["passed"]:
            print("采集质量提示：" + "；".join(report["issues"]))
            print("数据已保留，本阶段正常结束，可继续后续步骤。")
        else:
            print("采集质量检查通过。")
        record["quality_notes"] = report["issues"]
        print(f"分析报告：{report_path}")
        record["phase"] = "complete"
        if stage not in self.state["completed"]:
            self.state["completed"].append(stage)
        self.save()
        return True

    def prepare(self) -> bool:
        if __package__:
            from . import arm_excitation
        else:
            import arm_excitation
        print("\n第0步：FIGAROH 离线优化训练/验证激励轨迹，不需要串口。")
        directory, record = self.attempt("preparation")
        report_path = directory / "screening_report.json"
        if not report_path.exists():
            confirm("开始优化两条激励轨迹并检查整段运动？")
            record["phase"] = "screening"
            self.save()
            print("正在离线筛选，详细结果写入文件…")
            run_tool(arm_excitation, ["--output", str(directory)], directory / "details.log")
            record["phase"] = "captured"
            self.save()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("profile") == "figaroh_full_dynamic":
            print("训练/验证轨迹已独立优化并完成整段几何、位置、速度及加速度检查；不设力矩约束。")
            print("求解收敛状态与激励指标保存到报告，可行解不等于实测辨识已成功。")
        else:
            print("读取旧版筛选报告；完整动态辨识需要另外生成独立验证计划。")
        print("检查范围不含实物环境、线缆、相邻连杆和实时起始姿态，采集前仍需预览。")
        print(f"筛选报告：{report_path}")
        confirm("接受这组离线计划供后续阶段使用？")
        self.state["preparation"] = str(report_path.relative_to(self.output))
        record["phase"] = "complete"
        if "preparation" not in self.state["completed"]:
            self.state["completed"].append("preparation")
        self.save()
        return True

    def identify(self) -> bool:
        records = self.state['attempts']
        inputs = []
        for stage in ('dynamic', 'validation'):
            record = records.get(stage)
            if not record or record['phase'] not in ('captured', 'complete'):
                raise ValueError(f'请先完成{NAMES[stage]}；辨识阶段不会启动机械臂')
            capture = self.output / record['directory'] / 'capture.csv'
            if not capture.exists():
                raise ValueError(f'{NAMES[stage]}的原始数据不存在')
            inputs.append(capture)
        directory, record = self.attempt('identification')
        print(f'\n完整动态辨识：实际反馈处理 → 基参数求解 → 独立数据验证\n输出：{directory}')
        confirm('开始完整动态辨识？仅离线计算，不连接串口')
        if __package__:
            from . import arm_dynamics
        else:
            import arm_dynamics
        record['phase'] = 'fitting'
        self.save()
        run_tool(arm_dynamics, [str(inputs[0]), '--validation', str(inputs[1]),
                              '--output', str(directory / 'result')], directory / 'details.log')
        record['phase'] = 'complete'
        record['report'] = str((directory / 'result/report.json').relative_to(self.output))
        if 'identification' not in self.state['completed']:
            self.state['completed'].append('identification')
        self.save()
        print(f'求解及独立预测误差已保存：{directory / "result/report.md"}；不自动写入下位机。')
        return True

    def prepared_plan(self, stage: str, preparation: str | None = None) -> Path:
        saved = preparation or self.state.get("preparation")
        if not saved:
            raise ValueError("请先完成第0步仿真筛选；本阶段不会随机生成替代计划")
        path = self.output / saved
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("kind") != "arm_stage0_screening" or report.get("version") != 1:
            raise ValueError("第0步报告格式无效，请重新完成筛选")
        if report["urdf_sha256"] != arm_collect._sha256(arm_collect.URDF):
            raise ValueError("URDF 已变化，请重新完成第0步")
        if report.get("geometry_sha256", report["urdf_sha256"]) != arm_collect._sha256(arm_collect.URDF):
            raise ValueError("代理几何已变化，请重新完成第0步")
        if stage not in report['plans']:
            raise ValueError('当前第0步报告不包含该阶段计划；请重新生成训练/验证激励，静态采集仅用于旧计划')
        entry = report["plans"][stage]
        plan = path.parent / entry["file"]
        if arm_collect._sha256(plan) != entry["sha256"]:
            raise ValueError("第0步计划已变化，请重新完成筛选")
        return plan

    def run(self):
        print(f"\n会话目录：{self.output}")
        print("使用固定默认轨迹和采集设置；详细数据与日志保存在会话目录。")
        print("直接动态辨识：0 生成 → 1 训练采集 → 2 独立验证采集 → 3 完整求解；无需静态采集。")
        print("力矩直接使用原始反馈，不加载标定系数。")
        suggested = next((str(i) for i, s in enumerate(STAGES)
                          if s != 'static' and s not in self.state["completed"]), "q")
        while True:
            print("\n0. 激励生成  1. 训练采集  2. 验证采集  3. 完整辨识  4. 旧静态采集（可选）  q. 退出")
            choice = input(f"选择阶段 [{suggested}]：").strip().lower() or suggested
            if choice == "q":
                return
            if choice not in ("0", "1", "2", "3", "4"):
                print("请输入阶段序号或 q。")
                continue
            stage = STAGES[int(choice)]
            try:
                passed = self.run_stage(stage)
                suggested = str(int(choice) + 1) if passed and choice not in ("3", "4") else ("q" if passed else choice)
            except Cancelled:
                print("已取消本步，没有开始后续操作。已有文件保留，可稍后继续。")
            except KeyboardInterrupt:
                print("\n流程已中止；采集文件保留，可使用同一会话目录重新进入。")
                return
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                record = self.state["attempts"].get(stage)
                if record is not None:
                    with (self.output / record["directory"] / "details.log").open("a", encoding="utf-8") as stream:
                        traceback.print_exc(file=stream)
                    if record["phase"] == "captured" and isinstance(error, (ValueError, KeyError)):
                        record["phase"] = "rejected"
                        self.save()
                print(f"本阶段未完成：{error}")
                print(f"已有文件保留，详细日志位于 {self.output / stage}。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="机械臂 FIGAROH 激励生成、动态采集与完整辨识")
    parser.add_argument("--port", help="USB 串口；省略时交互选择")
    parser.add_argument(
        "--output", type=Path,
        default=Path.cwd() / "captures" / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f"),
        help="会话目录；使用已有目录可继续该会话",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        workflow = Workflow(args.output, args.port)
        workflow.run()
        print(f"已退出。文件保留在：{workflow.output}")
        return 0
    except (Cancelled, EOFError, KeyboardInterrupt):
        print("\n已退出，已有文件保留。")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"无法启动流程：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
