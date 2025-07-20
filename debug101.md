# วิธีการ Debug `RunLoader` เพื่อทำความเข้าใจการประมวลผลหลาย Worker

เอกสารนี้อธิบายวิธีการ debug การทำงานของ `RunLoader` ทั้งกระบวนการ ตั้งแต่การสแกนไดเรกทอรีที่มีไฟล์ trace จากหลาย worker, การแบ่ง worker/span, ไปจนถึงการประมวลผลข้อมูล distributed. วิธีนี้จะใช้สคริปต์แบบ Standalone เพื่อให้สามารถใช้ VS Code Debugger ไล่ดูการทำงานและค่าตัวแปรต่างๆ ได้อย่างละเอียด.

## ภาพรวมของแนวทาง

เราจะสร้างสคริปต์ Python (`debug_full_loader.py`) ที่จำลองการทำงานของเมธอด `RunLoader.load()` ทั้งหมด. ข้อแตกต่างที่สำคัญคือ เราจะเปลี่ยนส่วนที่ใช้ `multiprocessing` ให้เป็น `for` loop ธรรมดา (sequential) เพื่อให้ debugger สามารถติดตามการทำงานได้ใน process เดียว. วิธีนี้จะช่วยให้เราเห็นภาพรวมการไหลของข้อมูลทั้งหมดได้อย่างชัดเจน.

## ขั้นตอนการ Debug

### 1. สร้างไฟล์สคริปต์ `debug_full_loader.py`

สร้างไฟล์ใหม่ใน root directory ของโปรเจกต์ (ระดับเดียวกับ `tb_plugin/`) และตั้งชื่อว่า `debug_full_loader.py`. ใส่โค้ดทั้งหมดด้านล่างนี้ลงไป:

```python
import os
import sys
import bisect
from collections import defaultdict

# เพิ่ม path เพื่อให้สามารถ import tb_plugin ได้
# (อาจต้องปรับตามโครงสร้างโฟลเดอร์ของคุณ)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from tb_plugin.torch_tb_profiler import consts, io, utils
from tb_plugin.torch_tb_profiler.run import Run
from tb_plugin.torch_tb_profiler.profiler.data import RunProfileData, DistributedRunProfileData
from tb_plugin.torch_tb_profiler.profiler.run_generator import RunGenerator, DistributedRunGenerator
from tb_plugin.torch_tb_profiler.profiler.node import CommunicationNode

# ตั้งค่า logger เพื่อให้เห็น output ที่มีประโยชน์
logger = utils.get_logger()

# ==============================================================================
# ฟังก์ชันเลียนแบบ `RunLoader._process_distributed_profiles`
# ==============================================================================
def simulate_process_distributed_profiles(profiles: list[DistributedRunProfileData], span):
    logger.info("-> Entering simulate_process_distributed_profiles")
    has_communication = True
    comm_node_lists: list[list[CommunicationNode]] = []

    # --- จุดที่น่าสนใจ #6: ตรวจสอบข้อมูลดิบจากทุก worker ---
    # ตั้ง breakpoint ที่นี่เพื่อดู `profiles` ที่รับเข้ามา
    # ควรจะมีข้อมูลจาก 4 workers (ในกรณีตัวอย่าง)
    # ---------------------------------------------------------
    for data in profiles:
        logger.debug(f"Processing profile data for worker: {data.worker}")
        if data.has_communication and data.comm_node_list:
            comm_node_lists.append(data.comm_node_list)
            if len(comm_node_lists) > 1 and len(comm_node_lists[-1]) != len(comm_node_lists[0]):
                logger.error("Number of communication operation nodes don't match between workers")
                has_communication = False
        else:
            has_communication = False

    if not has_communication:
        logger.warning("Not all workers have communication data. Skipping distributed view.")
        return None

    # --- จุดที่น่าสนใจ #7: การ Synchronize เวลา Communication ---
    # นี่คือตรรกะที่สำคัญที่สุดในการทำความเข้าใจ distributed view
    # -------------------------------------------------------------
    worker_num = len(comm_node_lists)
    # วนลูปตาม communication op (เช่น all_reduce ครั้งที่ 1, 2, 3...)
    for i, node in enumerate(comm_node_lists[0]):
        kernel_range_size = len(node.kernel_ranges)
        # วนลูปตาม kernel ของ communication op นั้น
        for j in range(kernel_range_size):
            min_range = sys.maxsize
            # หา kernel ที่ทำงานสั้นที่สุดในบรรดา worker ทั้งหมดสำหรับ op เดียวกัน
            for k in range(worker_num):
                kernel_ranges = comm_node_lists[k][i].kernel_ranges
                if len(kernel_ranges) != kernel_range_size:
                    logger.error("Number of communication kernels don't match")
                    return None
                if kernel_ranges:
                    duration = kernel_ranges[j][1] - kernel_ranges[j][0]
                    if duration < min_range:
                        min_range = duration

            # --- จุดที่น่าสนใจ #8: ดู min_range และการคำนวณ real_time_ranges ---
            # ตั้ง breakpoint ที่นี่เพื่อดูค่า `min_range` ที่หาได้
            # และดูว่า `real_time_ranges` ของแต่ละ worker ถูกคำนวณใหม่ อย่างไร
            # ---------------------------------------------------------------------
            for k in range(worker_num):
                kernel_range = comm_node_lists[k][i].kernel_ranges[j]
                # real_time_ranges จะถูกใช้ใน DistributedRunGenerator เพื่อแยก Data Transfer Time และ Synchronizing Time
                comm_node_lists[k][i].real_time_ranges.append((kernel_range[1] - min_range, kernel_range[1]))

    # เรียก data.communication_parse() บนทุก worker อีกครั้ง
    # เพื่อคำนวณสถิติ communication โดยละเอียดโดยใช้ real_time_ranges ที่เพิ่งปรับแก้ไป
    for data in profiles:
        data.communication_parse()

    # --- จุดที่น่าสนใจ #9: ดูผลลัพธ์หลัง synchronize ---
    # ตั้ง breakpoint ที่นี่เพื่อดู `data.step_comm_stats` และ `data.total_comm_stats`
    # ที่คำนวณจากข้อมูลที่ synchronize แล้ว
    # -------------------------------------------------------------
    generator = DistributedRunGenerator(profiles, span)
    profile = generator.generate_run_profile()
    logger.info("<- Exiting simulate_process_distributed_profiles")
    return profile

# ==============================================================================
# ฟังก์ชันเลียนแบบ `RunLoader._process_spans`
# ==============================================================================
def simulate_process_spans(distributed_run: Run):
    logger.info("-> Entering simulate_process_spans")
    spans = distributed_run.get_spans()
    if spans is None:
        return [simulate_process_distributed_profiles(distributed_run.get_profiles(), None)]
    else:
        span_profiles = []
        for span in spans:
            profiles = distributed_run.get_profiles(span=span)
            p = simulate_process_distributed_profiles(profiles, span)
            if p is not None:
                span_profiles.append(p)
        return span_profiles

# ==============================================================================
# ฟังก์ชันเลียนแบบ `RunLoader._process_data`
# ==============================================================================
def simulate_process_data(worker, span, path, cache_dir):
    logger.info(f"-> Processing data for worker: {worker}")
    try:
        data = RunProfileData.parse(worker, span, path, cache_dir)
        generator = RunGenerator(worker, span, data)
        profile = generator.generate_run_profile()
        dist_data = DistributedRunProfileData(data)
        logger.info(f"<- Finished processing for worker: {worker}")
        return (profile, dist_data)
    except Exception as ex:
        logger.error(f"Failed to parse profile data for worker {worker}. Exception={ex}", exc_info=True)
        return (None, None)

# ==============================================================================
# ฟังก์ชันหลักที่เลียนแบบ `RunLoader.load`
# ==============================================================================
def simulate_run_loader_load(run_name, run_dir):
    logger.info("--- Starting Full Loader Simulation ---")

    # ส่วนที่ 1: ค้นหาและจัดระเบียบไฟล์ Trace
    workers = []
    spans_by_workers = defaultdict(list)
    for path in io.listdir(run_dir):
        if io.isdir(io.join(run_dir, path)):
            continue
        match = consts.WORKER_PATTERN.match(path)
        if not match:
            continue
        worker, span = match.group(1), match.group(2)
        if span is not None:
            span = span[1:]
            bisect.insort(spans_by_workers[worker], span)
        workers.append((worker, span, path))

    span_index_map = {}
    for worker, span_array in spans_by_workers.items():
        for i, span in enumerate(span_array, 1):
            span_index_map[(worker, span)] = i

    # --- จุดที่น่าสนใจ #1: ตรวจสอบไฟล์ที่พบ ---
    # ตั้ง breakpoint ที่นี่เพื่อดูค่าใน `workers` และ `span_index_map`
    # ---------------------------------------------
    logger.info(f"Found {len(workers)} worker files.")

    # ส่วนที่ 2: จำลอง Multiprocessing Loop (แบบ Sequential)
    queue_results = []
    cache_dir = os.path.join(os.path.dirname(run_dir), 'temp_cache')
    os.makedirs(cache_dir, exist_ok=True)

    for worker, span, path in workers:
        span_index = None if span is None else span_index_map.get((worker, span))

        # --- จุดที่น่าสนใจ #2: การประมวลผลแต่ละไฟล์ ---
        # ตั้ง breakpoint ที่บรรทัดถัดไปเพื่อไล่ดูการทำงานของ `simulate_process_data`
        # สำหรับไฟล์ trace แต่ละไฟล์
        # ----------------------------------------------------
        result = simulate_process_data(worker, span_index, os.path.join(run_dir, path), cache_dir)
        queue_results.append(result)

    # ส่วนที่ 3: รวบรวมผลลัพธ์
    distributed_run = Run(run_name, run_dir)
    run = Run(run_name, run_dir)

    for item in queue_results:
        r, d = item
        if r: run.add_profile(r)
        if d: distributed_run.add_profile(d)

    # --- จุดที่น่าสนใจ #3: ตรวจสอบข้อมูลหลังรวบรวม ---
    # ตั้ง breakpoint ที่นี่เพื่อดู `run` และ `distributed_run`
    # ว่ามี profile ครบตามจำนวน worker หรือไม่
    # ----------------------------------------------------
    logger.info(f"Aggregated {len(run.profiles)} profiles.")

    # ส่วนที่ 4: ประมวลผล Distributed View
    # --- จุดที่น่าสนใจ #4: ก่อนเข้าประมวลผล distributed ---
    # ตั้ง breakpoint ที่บรรทัดถัดไปเพื่อเข้าไปดูการทำงานของ
    # `simulate_process_spans` และ `simulate_process_distributed_profiles`
    # ---------------------------------------------------------
    distributed_profiles = simulate_process_spans(distributed_run)
    for d in distributed_profiles:
        if d: run.add_profile(d)

    # --- จุดที่น่าสนใจ #5: ตรวจสอบผลลัพธ์สุดท้าย ---
    # ตั้ง breakpoint ที่นี่เพื่อดู `run` object ที่เสร็จสมบูรณ์
    # ----------------------------------------------------
    logger.info("--- Loader Simulation Finished ---")
    return run

# ==============================================================================
# Entry Point ของสคริปต์
# ==============================================================================
if __name__ == "__main__":
    # กำหนด run ที่มีหลาย worker
    # คุณสามารถเปลี่ยนเป็น "resnet50_ddp" หรือไดเรกทอรีอื่นได้
    run_dir_name = "resnet50_ddp_4_workers"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    run_dir = os.path.join(script_dir, 'tb_plugin/samples/', run_dir_name)

    if not os.path.isdir(run_dir):
        print(f"Error: Directory not found at {run_dir}")
        print("Please make sure you are running this script from the root of the kineto repository.")
    else:
        final_run_object = simulate_run_loader_load(run_dir_name, run_dir)
        print(f"\n--- Final Result ---")
        print(f"Total profiles loaded into final Run object: {len(final_run_object.profiles)}")
        # ตั้ง breakpoint ที่นี่เพื่อสำรวจ final_run_object
        pass

```

### 2. วิธีการใช้งานและจุดที่น่าสนใจในการ Debug

1.  **บันทึกไฟล์**: บันทึกโค้ดข้างต้นเป็นไฟล์ `debug_full_loader.py` ใน root directory ของ repository.

2.  **เปิดไฟล์ใน VS Code**: เปิดไฟล์ `debug_full_loader.py` ใน VS Code.

3.  **ตั้ง Breakpoint ในจุดที่น่าสนใจ**:
    *   **จุดที่ #1**: ดูผลลัพธ์ของการสแกนไฟล์.
    *   **จุดที่ #2**: ไล่ดูการประมวลผลไฟล์ trace แต่ละไฟล์. สามารถ Step Into (`F11`) เข้าไปใน `simulate_process_data` เพื่อดูการทำงานของ `RunProfileData.parse()` และ parser อื่นๆ สำหรับไฟล์นั้นๆ.
    *   **จุดที่ #3**: ตรวจสอบว่าข้อมูลจากทุกไฟล์ถูกรวบรวมมาถูกต้องหรือไม่.
    *   **จุดที่ #4**: จุดเปลี่ยนสำคัญก่อนจะเข้าไปในตรรกะของ distributed view.
    *   **จุดที่ #5**: ดู `Run` object สุดท้ายที่เสร็จสมบูรณ์.
    *   **จุดที่ #6**: (ใน `simulate_process_distributed_profiles`) ดูข้อมูลดิบของแต่ละ worker ก่อนทำการ synchronize.
    *   **จุดที่ #7, #8**: **ส่วนที่สำคัญที่สุด** ในการทำความเข้าใจตรรกะ distributed. ไล่ดู loop นี้เพื่อดูว่า `min_range` ถูกคำนวณอย่างไร และ `real_time_ranges` ถูกสร้างขึ้นมาใหม่ได้อย่างไร.
    *   **จุดที่ #9**: ดูผลลัพธ์ทางสถิติของ communication หลังจากข้อมูลถูก synchronize แล้ว.

4.  **เริ่ม Debug**: กด **F5** เพื่อเริ่มการ debug. โปรแกรมจะหยุดที่ breakpoint แรกที่คุณตั้งไว้. จากนั้นคุณสามารถ:
    *   **Step Over (F10)**: รันบรรทัดปัจจุบันและไปบรรทัดถัดไป.
    *   **Step Into (F11)**: เข้าไปดูการทำงานภายในฟังก์ชันที่กำลังจะเรียก.
    *   **Step Out (Shift+F11)**: รันโค้ดที่เหลือในฟังก์ชันปัจจุบันให้เสร็จและกลับออกมา.
    *   **ดูค่าตัวแปร**: ใช้หน้าต่าง "VARIABLES" ทางด้านซ้าย หรือเอาเมาส์ไปชี้บนตัวแปรในโค้ดเพื่อดูค่าปัจจุบัน.

ด้วยสคริปต์และขั้นตอนเหล่านี้ คุณจะสามารถไล่การทำงานของ `RunLoader` ทั้งหมดและทำความเข้าใจการไหลของข้อมูลตั้งแต่ต้นจนจบได้อย่างละเอียด.
