# วิธีการดึง Operator Tree (`tid2tree`) จากไฟล์ Trace โดยตรง

เอกสารนี้แสดงวิธีการเขียนสคริปต์ Python เพื่อดึงข้อมูล `tid2tree` และ `pl_tid2tree` (โครงสร้าง Operator Tree) จากไฟล์ `.pt.trace.json.gz` โดยตรง โดยใช้ส่วนประกอบที่จำเป็นจาก `torch-tb-profiler` และตัดส่วนที่ไม่เกี่ยวข้องออกไป เพื่อให้ได้โค้ดที่กระชับและตรงตามเป้าหมาย.

## 1. ส่วนประกอบที่จำเป็น

ในการที่จะสร้าง `tid2tree` ได้, คุณจำเป็นต้องมีไฟล์ Python ต่อไปนี้จาก `tb_plugin/torch_tb_profiler/profiler/` ในโปรเจกต์ของคุณ และต้องรักษโครงสร้างโฟลเดอร์ไว้เพื่อให้ import ทำงานได้ถูกต้อง:

*   `profiler/trace.py`: นิยามโครงสร้าง Event และ factory.
*   `profiler/node.py`: นิยามโครงสร้าง Node (เช่น `OperatorNode`).
*   `profiler/range_utils.py`: Utilities สำหรับจัดการช่วงเวลา.
*   `profiler/op_tree.py`: `OpTreeBuilder` สำหรับสร้าง Tree.
*   `profiler/communication.py`: `generate_communication_nodes` (ถูกเรียกโดย `EventParser`).
*   `profiler/event_parser.py`: `EventParser` ซึ่งเป็น Engine หลักในการประมวลผล.

**ไฟล์ที่ไม่จำเป็น** สำหรับเป้าหมายนี้คือ: `loader.py`, `data.py`, `run_generator.py`, `op_agg.py`, และ parser เฉพาะทางอื่นๆ (`OverallParser`, `GPUMetricsParser` ฯลฯ).

## 2. โค้ดตัวอย่าง (`get_op_tree.py`)

สร้างไฟล์ใหม่ชื่อ `get_op_tree.py` ในระดับเดียวกับโฟลเดอร์ `tb_plugin/` และใส่โค้ดต่อไปนี้:

```python
import os
import sys
import gzip
import json
from json.decoder import JSONDecodeError
import io as sysio
import re

# เพิ่ม path เพื่อให้สามารถ import tb_plugin ได้
# (อาจต้องปรับตามโครงสร้างโฟลเดอร์ของคุณ)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import ส่วนประกอบที่จำเป็น
from tb_plugin.torch_tb_profiler.profiler.trace import create_event, create_association_events
from tb_plugin.torch_tb_profiler.profiler.event_parser import EventParser

def load_trace_json(trace_path: str) -> dict:
    """
    ฟังก์ชันสำหรับโหลดและทำความสะอาดไฟล์ trace.
    ยกตรรกะบางส่วนมาจาก RunProfileData._preprocess_file.
    """
    print(f"Loading trace file: {trace_path}")
    try:
        with gzip.open(trace_path, 'rb') as f:
            trace_data = f.read()
    except Exception as e:
        print(f"Failed to read or decompress trace file: {e}")
        return None

    # จัดการกับ JSON ที่อาจมี format ไม่ถูกต้อง
    try:
        trace_json = json.loads(trace_data)
    except JSONDecodeError:
        print("JSONDecodeError found. Trying to fix and re-parse...")
        try:
            # พยายาม re-encode และแก้ไข "N/A" ที่ไม่มี quote
            str_data = trace_data.decode('utf-8')
            fixed_str_data = re.sub(r'(?<!")N/A(?!")', "\"N/A\"", str_data)
            trace_json = json.loads(fixed_str_data)
        except Exception as e:
            print(f"Failed to fix and re-parse JSON: {e}")
            return None

    # สามารถเพิ่มตรรกะการลบ 'Record Window End' event ได้ที่นี่ถ้าจำเป็น
    # ...

    print("Trace file loaded and parsed successfully.")
    return trace_json

def parse_trace_to_tree(trace_json: dict) -> tuple:
    """
    ฟังก์ชันหลักที่รับ trace_json และคืนค่า tid2tree.
    เลียนแบบการทำงานของ RunProfileData.__init__ และ process() เฉพาะส่วนที่จำเป็น.
    """
    if not trace_json or 'traceEvents' not in trace_json:
        print("Invalid trace_json format.")
        return None, None

    # 1. เลียนแบบ RunProfileData.__init__: สร้าง event objects และ fwd/bwd map
    print("Creating event objects...")
    trace_body = trace_json.get('traceEvents', [])
    events = []
    fwd_bwd_events = []
    is_pytorch_lightning = trace_json.get('Framework', None) == 'pytorch-lightning'

    for data in trace_body:
        if data.get('cat') == 'fwdbwd':
            fwd_bwd_events.append(data)
        else:
            event = create_event(data, is_pytorch_lightning)
            if event:
                events.append(event)

    events.sort(key=lambda e: e.ts)
    fwd_bwd_map = create_association_events(fwd_bwd_events)
    print(f"Created {len(events)} event objects.")
    print(f"Found {len(fwd_bwd_map)} forward-backward associations.")

    # 2. เลียนแบบ RunProfileData.process: เรียกใช้ EventParser
    print("Running EventParser to build operator tree...")
    parser = EventParser()
    tid2tree, pl_tid2tree = parser.parse(events, fwd_bwd_map)
    print("EventParser finished.")

    return tid2tree, pl_tid2tree

if __name__ == "__main__":
    # 3. กำหนด path ไปยังไฟล์ trace และเรียกใช้งานฟังก์ชัน
    script_dir = os.path.dirname(os.path.realpath(__file__))
    trace_path = os.path.join(script_dir, 'tb_plugin/samples/resnet50_ddp_4_workers/worker0.1623143089861.pt.trace.json.gz')

    if not os.path.exists(trace_path):
        print(f"Error: Trace file not found at {trace_path}")
    else:
        # โหลดไฟล์ trace
        trace_json_data = load_trace_json(trace_path)

        if trace_json_data:
            # ประมวลผลเพื่อเอา Operator Tree
            tid2tree, pl_tid2tree = parse_trace_to_tree(trace_json_data)

            # 4. ตรวจสอบผลลัพธ์
            if tid2tree:
                print("\n--- Operator Tree (tid2tree) Result ---")
                for tid, root_node in tid2tree.items():
                    print(f"  Thread ID: {tid}")
                    print(f"  Root node name: {root_node.name}")
                    print(f"  Number of children at root: {len(root_node.children)}")
                    # คุณสามารถตั้ง breakpoint ที่นี่เพื่อสำรวจ `root_node` และโครงสร้าง tree ภายใน
                    # ตัวอย่าง: พิมพ์ชื่อ child แรก
                    if root_node.children:
                        print(f"  First child: {root_node.children[0].name}")

            if pl_tid2tree:
                print("\n--- PyTorch Lightning Module Tree (pl_tid2tree) Result ---")
                # ... (ตรรกะคล้ายกันสำหรับ pl_tid2tree) ...

```

## 3. วิธีการใช้งาน

1.  **จัดโครงสร้างไฟล์**: ตรวจสอบให้แน่ใจว่าคุณได้คัดลอกไฟล์ที่จำเป็น 6 ไฟล์ (`trace.py`, `node.py`, `range_utils.py`, `op_tree.py`, `communication.py`, `event_parser.py`) มาไว้ในโครงสร้างโฟลเดอร์ที่ถูกต้อง (`your_project/tb_plugin/torch_tb_profiler/profiler/`).
2.  **วางสคริปต์**: วางไฟล์ `get_op_tree.py` ไว้ใน `your_project/`.
3.  **แก้ไข Path (ถ้าจำเป็น)**: ใน `get_op_tree.py`, แก้ไข `trace_path` ให้ชี้ไปยังไฟล์ `.pt.trace.json.gz` ที่คุณต้องการจะวิเคราะห์.
4.  **รันสคริปต์**:
    ```bash
    python get_op_tree.py
    ```

## 4. ผลลัพธ์ที่ได้ (`tid2tree`)

สคริปต์จะคืนค่า `tid2tree` ซึ่งเป็น dictionary ของ Python.

*   **Key**: `int` - คือ Thread ID (TID) ของแต่ละ thread ที่มีการทำงาน.
*   **Value**: `OperatorNode` - คือ object ที่เป็น root ของ operator tree สำหรับ thread นั้นๆ.

`OperatorNode` object (นิยามใน `profiler/node.py`) มี attributes ที่สำคัญคือ:
*   `node.name`: `str` - ชื่อของ operator.
*   `node.start_time`, `node.end_time`: `int` - เวลาเริ่มต้นและสิ้นสุด.
*   `node.children`: `List[OperatorNode]` - **list ของ child nodes**. นี่คือส่วนที่คุณสามารถใช้เพื่อ traverse tree ต่อไปได้.
*   `node.runtimes`: `List[RuntimeNode]` - list ของ runtime events ที่เกี่ยวข้องกับ operator นี้.
*   `node.device_duration`, `node.self_device_duration`: `int` - เวลาที่ใช้บน device.
*   และ attributes อื่นๆ อีกมากมาย.

คุณสามารถเขียนฟังก์ชัน recursive เพื่อวนลูปผ่าน `node.children` ของแต่ละ node เพื่อสำรวจโครงสร้าง tree ทั้งหมดและดึงข้อมูลที่คุณต้องการออกมาได้.
