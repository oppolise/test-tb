# เอกสารอธิบายไฟล์ consts.py

ไฟล์ `consts.py` ใน `torch_tb_profiler` นี้เป็นศูนย์รวมค่าคงที่ต่างๆ ที่ใช้ภายในโปรไฟล์เลอร์ โดยค่าคงที่เหล่านี้ช่วยให้การจัดการและการอ้างอิงค่าต่างๆ เป็นไปอย่างมีระบบและง่ายต่อการแก้ไขในอนาคต

## การนำเข้าโมดูล (Import Modules)

```python
import re
from collections import namedtuple
```

- `import re`: นำเข้าโมดูล `re` ซึ่งเป็นโมดูลสำหรับการทำงานกับ Regular Expressions ใช้สำหรับการจับคู่และค้นหารูปแบบข้อความที่ซับซ้อน เช่น การแยกชื่อ worker หรือ node process
- `from collections import namedtuple`: นำเข้า `namedtuple` จากโมดูล `collections` ซึ่งเป็น factory function สำหรับสร้าง subclass ของ tuple ที่มีชื่อฟิลด์ ทำให้สามารถเข้าถึงสมาชิกของ tuple ด้วยชื่อได้ เพิ่มความชัดเจนในการอ่านโค้ด

## ค่าคงที่หลัก (Main Constants)

```python
PLUGIN_NAME = 'pytorch_profiler'
```

- `PLUGIN_NAME`: เป็นสตริงที่กำหนดชื่อของปลั๊กอินนี้คือ `'pytorch_profiler'`
    - **ทำไมต้องกำหนด**: ใช้เพื่อระบุชื่อของปลั๊กอินนี้ใน TensorBoard ทำให้ TensorBoard รู้จักและโหลดปลั๊กอินนี้ได้อย่างถูกต้อง
    - **การใช้งาน**: อาจถูกใช้ในการลงทะเบียนปลั๊กอินกับ TensorBoard หรือใช้เป็นส่วนหนึ่งของ namespace หรือ prefix ในการตั้งชื่อ log หรือข้อมูลต่างๆ ที่เกี่ยวข้องกับโปรไฟล์เลอร์นี้

## รูปแบบ Regular Expression (Regular Expression Patterns)

```python
WORKER_PATTERN = re.compile(r"""^(.*?) # worker name
        (\.\d+)? # optional timestamp like 1619499959628 used as span name
        \.pt\.trace\.json # the ending suffix
        (?:\.gz)?$""", re.X)  # optional .gz extension
```

- `WORKER_PATTERN`: เป็นออบเจ็กต์ regular expression ที่คอมไพล์แล้ว ใช้สำหรับจับคู่และแยกส่วนประกอบของชื่อไฟล์ trace ของ PyTorch
    - **คำอธิบายรูปแบบ**:
        - `^(.*?)`: ส่วนแรกสุดของสตริง (^) คือชื่อ worker (captured group ที่ 1) โดย `(.*?)` หมายถึงอักขระใดๆ ศูนย์ตัวหรือมากกว่า แบบไม่โลภ (non-greedy)
        - `(\.\d+)?`: ส่วนนี้เป็น optional (เครื่องหมาย `?` ต่อท้าย) คือ timestamp ที่อาจมีหรือไม่มีก็ได้ (captured group ที่ 2) โดย `\.` คือจุดทศนิยม และ `\d+` คือตัวเลขหนึ่งตัวหรือมากกว่า
        - `\.pt\.trace\.json`: ส่วนนี้คือส่วนท้ายของชื่อไฟล์ที่ต้องมีเสมอ คือ `.pt.trace.json`
        - `(?:\.gz)?$`: ส่วนนี้เป็น optional non-capturing group `(?: ... )` คือนามสกุล `.gz` ที่อาจมีหรือไม่มีก็ได้ และ `$` หมายถึงจุดสิ้นสุดของสตริง
        - `re.X`: flag นี้อนุญาตให้ใส่คอมเมนต์และจัดรูปแบบ regular expression ให้อ่านง่ายขึ้น
    - **ทำไมต้องกำหนด**: ใช้ในการแยกชื่อ worker และข้อมูลอื่นๆ จากชื่อไฟล์ trace ที่โปรไฟล์เลอร์สร้างขึ้น ทำให้สามารถระบุได้ว่า trace นี้มาจาก worker ใด และอาจมีข้อมูล timestamp เพิ่มเติมหรือไม่
    - **การใช้งาน**: เมื่อโปรแกรมได้รับรายการไฟล์ trace จะใช้ pattern นี้เพื่อตรวจสอบและดึงข้อมูลที่สำคัญออกจากชื่อไฟล์

```python
NODE_PROCESS_PATTERN = re.compile(r"""^(.*)_(\d+)""")
```

- `NODE_PROCESS_PATTERN`: เป็นออบเจ็กต์ regular expression ที่คอมไพล์แล้ว ใช้สำหรับจับคู่และแยกชื่อโหนด (node) และหมายเลขโปรเซส (process ID)
    - **คำอธิบายรูปแบบ**:
        - `^(.*)`: ส่วนแรกสุดของสตริง (^) คือชื่อโหนด (captured group ที่ 1)
        - `_`: ตัวคั่น underscore
        - `(\d+)`: หมายเลขโปรเซส (captured group ที่ 2) ซึ่งเป็นตัวเลขหนึ่งตัวหรือมากกว่า
    - **ทำไมต้องกำหนด**: ในระบบ distributed training ชื่อของ worker อาจมีรูปแบบที่รวมชื่อโหนดและ ID ของโปรเซสไว้ด้วยกัน pattern นี้ช่วยในการแยกข้อมูลทั้งสองส่วนออกจากกัน
    - **การใช้งาน**: ใช้ในการประมวลผลชื่อ worker เพื่อระบุโหนดและโปรเซสที่เกี่ยวข้อง

## ค่าคงที่สำหรับการทำงานของ Monitor

```python
MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS = 10
```

- `MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS`: เป็นจำนวนเต็มที่กำหนดช่วงเวลา (ในหน่วยวินาที) ในการรีเฟรชข้อมูลของ monitor run คือ `10` วินาที
    - **ทำไมต้องกำหนด**: ใช้ควบคุมความถี่ในการตรวจสอบหรืออัปเดตข้อมูลล่าสุดจากโปรไฟล์เลอร์เมื่อมีการทำงานในโหมด monitor
    - **การใช้งาน**: ระบบ monitor จะใช้ค่านี้เพื่อตั้งเวลาในการดึงข้อมูลใหม่ ทำให้ผู้ใช้เห็นข้อมูลที่ค่อนข้างล่าสุดโดยไม่ต้องรีเฟรชเองบ่อยๆ

```python
MAX_GPU_PER_NODE = 64
```

- `MAX_GPU_PER_NODE`: เป็นจำนวนเต็มที่กำหนดจำนวน GPU สูงสุดต่อโหนด คือ `64`
    - **ทำไมต้องกำหนด**: อาจใช้เป็นค่าจำกัดหรือค่าอ้างอิงในการจัดสรรทรัพยากรหรือแสดงข้อมูลที่เกี่ยวข้องกับ GPU ในแต่ละโหนด
    - **การใช้งาน**: อาจใช้ในการสร้างโครงสร้างข้อมูลที่เก็บข้อมูล GPU หรือในการตรวจสอบความถูกต้องของจำนวน GPU ที่รายงาน

## การกำหนด View (Views Definition)

```python
View = namedtuple('View', 'id, name, display_name')
OVERALL_VIEW = View(1, 'overall', 'Overview')
OP_VIEW = View(2, 'operator', 'Operator')
KERNEL_VIEW = View(3, 'kernel', 'Kernel')
TRACE_VIEW = View(4, 'trace', 'Trace')
DISTRIBUTED_VIEW = View(5, 'distributed', 'Distributed')
MEMORY_VIEW = View(6, 'memory', 'Memory')
MODULE_VIEW = View(7, 'module', 'Module')
LIGHTNING_VIEW = View(8, 'lightning', 'Lightning')
```

- `View = namedtuple('View', 'id, name, display_name')`: สร้างคลาส `View` ใหม่โดยใช้ `namedtuple` ซึ่งแต่ละ instance ของ `View` จะมี 3 ฟิลด์คือ `id` (ตัวระบุที่เป็นตัวเลข), `name` (ชื่อภายในที่ใช้ในโค้ด), และ `display_name` (ชื่อที่แสดงผลให้ผู้ใช้เห็น)
    - **ทำไมต้องกำหนด**: เพื่อจัดระเบียบและกำหนดประเภทของมุมมอง (view) ต่างๆ ที่โปรไฟล์เลอร์จะแสดงผลใน TensorBoard ทำให้โค้ดที่เกี่ยวข้องกับการจัดการ view อ่านง่ายและเป็นระบบ
    - **การใช้งาน**: ค่า `View` เหล่านี้จะถูกใช้ในการสร้างแท็บหรือส่วนต่างๆ ในหน้า UI ของโปรไฟล์เลอร์ใน TensorBoard
        - `OVERALL_VIEW`: มุมมองภาพรวม (Overview)
        - `OP_VIEW`: มุมมองระดับ Operator (Operator View)
        - `KERNEL_VIEW`: มุมมองระดับ Kernel (Kernel View)
        - `TRACE_VIEW`: มุมมอง Trace (Trace View) สำหรับการแสดงผล trace events
        - `DISTRIBUTED_VIEW`: มุมมองสำหรับการวิเคราะห์ distributed training
        - `MEMORY_VIEW`: มุมมองสำหรับการวิเคราะห์การใช้หน่วยความจำ
        - `MODULE_VIEW`: มุมมองสำหรับการวิเคราะห์ตามโมดูลของ PyTorch (เช่น `nn.Module`)
        - `LIGHTNING_VIEW`: มุมมองเฉพาะสำหรับ PyTorch Lightning

## คำอธิบาย Tooltips (Tooltip Constants)

ส่วนนี้จะกำหนดข้อความที่จะแสดงเป็น tooltip (คำแนะนำที่ปรากฏเมื่อนำเมาส์ไปชี้) สำหรับเมตริกต่างๆ ใน UI ของโปรไฟล์เลอร์ การกำหนดเป็นค่าคงที่ช่วยให้จัดการข้อความได้ง่ายและสอดคล้องกัน

```python
TOOLTIP_GPU_UTIL = \
    'GPU Utilization:\n' \
    'GPU busy time / All steps time. The higher, the better. ' \
    'GPU busy time is the time during which there is at least one GPU kernel running on it. ' \
    'All steps time is the total time of all profiler steps(or called as iterations).\n'
```

- `TOOLTIP_GPU_UTIL`: ข้อความอธิบายสำหรับเมตริก "GPU Utilization"
    - **คำอธิบาย**: อธิบายว่า GPU Utilization คืออะไร (เวลาที่ GPU ไม่ว่าง / เวลาทั้งหมดของทุกขั้นตอน), ยิ่งสูงยิ่งดี, เวลาที่ GPU ไม่ว่างคืออะไร, และเวลาทั้งหมดของทุกขั้นตอนคืออะไร
    - **ทำไมต้องกำหนด**: เพื่อให้ผู้ใช้เข้าใจความหมายของเมตริก GPU Utilization และวิธีการคำนวณ
    - **การใช้งาน**: แสดงใน UI เมื่อผู้ใช้ชี้เมาส์ไปที่ส่วนที่แสดง GPU Utilization

```python
TOOLTIP_SM_EFFICIENCY = \
    'Est. SM Efficiency:\n' \
    'Estimated Stream Multiprocessor Efficiency. The higher, the better. ' \
    'This metric of a kernel, SM_Eff_K = min(blocks of this kernel / SM number of this GPU, 100%). ' \
    "This overall number is the sum of all kernels' SM_Eff_K weighted by kernel's execution duration, " \
    'divided by all steps time.\n'
```

- `TOOLTIP_SM_EFFICIENCY`: ข้อความอธิบายสำหรับเมตริก "Est. SM Efficiency" (ประสิทธิภาพโดยประมาณของ Stream Multiprocessor)
    - **คำอธิบาย**: อธิบายว่า SM Efficiency คืออะไร, ยิ่งสูงยิ่งดี, สูตรการคำนวณสำหรับ kernel (SM_Eff_K), และวิธีการคำนวณค่าโดยรวม (ผลรวมถ่วงน้ำหนักของ SM_Eff_K ของทุก kernel หารด้วยเวลาทั้งหมดของทุกขั้นตอน)
    - **ทำไมต้องกำหนด**: เพื่อให้ผู้ใช้เข้าใจความหมายของ SM Efficiency และวิธีการคำนวณ
    - **การใช้งาน**: แสดงใน UI เมื่อผู้ใช้ชี้เมาส์ไปที่ส่วนที่แสดง SM Efficiency

```python
TOOLTIP_OCCUPANCY_COMMON = \
    'Est. Achieved Occupancy:\n' \
    'For most cases such as memory bandwidth bounded kernels, the higher the better. ' \
    'Occupancy is the ratio of active warps on an SM ' \
    'to the maximum number of active warps supported by the SM. ' \
    'The theoretical occupancy of a kernel is upper limit occupancy of this kernel, ' \
    'limited by multiple factors such as kernel shape, kernel used resource, ' \
    'and the GPU compute capability.\n' \
    'Est. Achieved Occupancy of a kernel, OCC_K = ' \
    'min(threads of the kernel / SM number / max threads per SM, theoretical occupancy of the kernel). '
```

- `TOOLTIP_OCCUPANCY_COMMON`: ข้อความอธิบายทั่วไปสำหรับเมตริก "Est. Achieved Occupancy" (ค่า Occupancy ที่ทำได้โดยประมาณ)
    - **คำอธิบาย**: อธิบายว่า Occupancy คืออะไร (อัตราส่วนของ active warps ต่อจำนวน active warps สูงสุดที่ SM รองรับ), โดยทั่วไปยิ่งสูงยิ่งดี, Theoretical Occupancy คืออะไร, และสูตรการคำนวณ Est. Achieved Occupancy ของ kernel (OCC_K)
    - **ทำไมต้องกำหนด**: เพื่อให้ผู้ใช้เข้าใจแนวคิดพื้นฐานของ Occupancy และปัจจัยที่เกี่ยวข้อง
    - **การใช้งาน**: เป็นส่วนหนึ่งของคำอธิบาย Occupancy ที่อาจใช้ร่วมกับ `TOOLTIP_OCCUPANCY_OVERVIEW` หรือ `TOOLTIP_OCCUPANCY_TABLE`

```python
TOOLTIP_OCCUPANCY_OVERVIEW = \
    "This overall number is the weighted average of all kernels' OCC_K " \
    "using kernel's execution duration as weight. " \
    'It shows fine-grained low-level GPU utilization.\n'
```

- `TOOLTIP_OCCUPANCY_OVERVIEW`: ข้อความอธิบายเพิ่มเติมสำหรับ "Est. Achieved Occupancy" ในมุมมองภาพรวม
    - **คำอธิบาย**: อธิบายว่าค่า Occupancy โดยรวมคำนวณมาจากการถ่วงน้ำหนัก OCC_K ของทุก kernel ด้วยระยะเวลาการทำงานของ kernel นั้นๆ และแสดงให้เห็นถึงการใช้งาน GPU ในระดับต่ำอย่างละเอียด
    - **ทำไมต้องกำหนด**: เพื่อให้คำอธิบายที่เฉพาะเจาะจงสำหรับค่า Occupancy ที่แสดงในส่วนภาพรวม
    - **การใช้งาน**: แสดงใน UI ของส่วนภาพรวมเมื่อผู้ใช้ชี้เมาส์ไปที่เมตริก Occupancy

```python
TOOLTIP_TENSOR_CORES = \
    'Kernel using Tensor Cores:\n' \
    'Total GPU Time for Tensor Core kernels / Total GPU Time for all kernels.\n'
```

- `TOOLTIP_TENSOR_CORES`: ข้อความอธิบายสำหรับเมตริก "Kernel using Tensor Cores"
    - **คำอธิบาย**: อธิบายว่าเมตริกนี้คือสัดส่วนของเวลา GPU ทั้งหมดที่ใช้โดย kernel ที่ทำงานบน Tensor Cores เทียบกับเวลา GPU ทั้งหมดของทุก kernel
    - **ทำไมต้องกำหนด**: เพื่อให้ผู้ใช้เข้าใจว่าส่วนใดของ GPU workload ที่กำลังใช้ประโยชน์จาก Tensor Cores
    - **การใช้งาน**: แสดงใน UI เมื่อผู้ใช้ชี้เมาส์ไปที่ส่วนที่แสดงการใช้งาน Tensor Cores

```python
TOOLTIP_OCCUPANCY_TABLE = \
    "This \"Mean\" number is the weighted average of all calls' OCC_K of the kernel, " \
    "using each call's execution duration as weight. " \
    'It shows fine-grained low-level GPU utilization.'
```

- `TOOLTIP_OCCUPANCY_TABLE`: ข้อความอธิบายสำหรับ "Mean Est. Achieved Occupancy" ในตารางข้อมูล (เช่น ตาราง Operator หรือ Kernel)
    - **คำอธิบาย**: อธิบายว่าค่า "Mean" Occupancy สำหรับ kernel หรือ operator นั้นคำนวณมาจากการถ่วงน้ำหนัก OCC_K ของการเรียกแต่ละครั้งด้วยระยะเวลาการทำงานของการเรียกนั้นๆ
    - **ทำไมต้องกำหนด**: เพื่อให้คำอธิบายที่เฉพาะเจาะจงสำหรับค่า Occupancy ที่แสดงในรูปแบบตาราง
    - **การใช้งาน**: แสดงใน UI ของตารางข้อมูลเมื่อผู้ใช้ชี้เมาส์ไปที่คอลัมน์ Occupancy

```python
TOOLTIP_BLOCKS_PER_SM = \
    'Blocks Per SM = blocks of this kernel / SM number of this GPU.\n' \
    'If this number is less than 1, it indicates the GPU multiprocessors are not fully utilized.\n' \
    '\"Mean Blocks per SM\" is the weighted average of all calls of this kernel, ' \
    "using each call's execution duration as weight."
```

- `TOOLTIP_BLOCKS_PER_SM`: ข้อความอธิบายสำหรับเมตริก "Blocks Per SM"
    - **คำอธิบาย**: อธิบายสูตรการคำนวณ Blocks Per SM, ความหมายถ้าค่าน้อยกว่า 1 (GPU multiprocessors ไม่ได้ถูกใช้งานเต็มที่), และวิธีการคำนวณ "Mean Blocks per SM" (ค่าเฉลี่ยถ่วงน้ำหนัก)
    - **ทำไมต้องกำหนด**: เพื่อช่วยผู้ใช้ประเมินว่า kernel ที่เรียกใช้งานนั้นสามารถใช้ประโยชน์จาก SM ทั้งหมดบน GPU ได้ดีเพียงใด
    - **การใช้งาน**: แสดงใน UI เมื่อผู้ใช้ชี้เมาส์ไปที่เมตริก Blocks Per SM

```python
TOOLTIP_OP_TC_ELIGIBLE = \
    'Whether this operator is eligible to use Tensor Cores.'
```

- `TOOLTIP_OP_TC_ELIGIBLE`: ข้อความอธิบายสำหรับสถานะ "eligibility" ของ Operator ในการใช้ Tensor Cores
    - **คำอธิบาย**: บอกว่า Operator นี้มีคุณสมบัติเหมาะสมที่จะใช้ Tensor Cores หรือไม่
    - **ทำไมต้องกำหนด**: เพื่อให้ข้อมูลเพิ่มเติมว่าทำไม Operator บางตัวถึงใช้หรือไม่ใช้ Tensor Cores
    - **การใช้งาน**: แสดงในตาราง Operator เพื่อระบุว่า Operator นั้นๆ สามารถใช้ Tensor Cores ได้หรือไม่

```python
TOOLTIP_OP_TC_SELF = \
    'Time of self-kernels with Tensor Cores / Time of self-kernels.'
```

- `TOOLTIP_OP_TC_SELF`: ข้อความอธิบายสำหรับเมตริกสัดส่วนเวลาการใช้ Tensor Cores ของ "self-kernels" ใน Operator
    - **คำอธิบาย**: สัดส่วนของเวลาที่ "self-kernels" (kernel ที่ถูกเรียกโดยตรงจาก Operator นั้นๆ ไม่รวม kernel จาก child operator) ใช้ Tensor Cores เทียบกับเวลาทั้งหมดของ self-kernels
    - **ทำไมต้องกำหนด**: เพื่อช่วยในการวิเคราะห์ว่าส่วน self-execution ของ Operator ได้ใช้ Tensor Cores มากน้อยเพียงใด
    - **การใช้งาน**: แสดงในมุมมอง Operator เพื่อให้รายละเอียดเกี่ยวกับการใช้ Tensor Cores ภายใน Operator เอง

```python
TOOLTIP_OP_TC_TOTAL = \
    'Time of kernels with Tensor Cores / Time of kernels.'
```

- `TOOLTIP_OP_TC_TOTAL`: ข้อความอธิบายสำหรับเมตริกสัดส่วนเวลาการใช้ Tensor Cores ของ kernel ทั้งหมด (รวม child operator) ใน Operator
    - **คำอธิบาย**: สัดส่วนของเวลาที่ kernel ทั้งหมด (รวม kernel จาก child operator) ที่เกี่ยวข้องกับ Operator นี้ใช้ Tensor Cores เทียบกับเวลาทั้งหมดของ kernel เหล่านั้น
    - **ทำไมต้องกำหนด**: เพื่อให้เห็นภาพรวมการใช้ Tensor Cores ของ Operator และทุกสิ่งที่อยู่ภายใต้ Operator นั้น
    - **การใช้งาน**: แสดงในมุมมอง Operator เพื่อให้ภาพรวมการใช้ Tensor Cores

```python
TOOLTIP_KERNEL_USES_TC = \
    'Whether this kernel uses Tensor Cores.'
```

- `TOOLTIP_KERNEL_USES_TC`: ข้อความอธิบายสำหรับสถานะการใช้ Tensor Cores ของ Kernel
    - **คำอธิบาย**: บอกว่า Kernel นี้ใช้ Tensor Cores หรือไม่
    - **ทำไมต้องกำหนด**: เพื่อให้ข้อมูลโดยตรงว่า Kernel แต่ละตัวมีการใช้ Tensor Cores จริงหรือไม่
    - **การใช้งาน**: แสดงในตาราง Kernel

```python
TOOLTIP_KERNEL_OP_TC_ELIGIBLE = \
    'Whether the operator launched this kernel is eligible to use Tensor Cores.'
```

- `TOOLTIP_KERNEL_OP_TC_ELIGIBLE`: ข้อความอธิบายสำหรับสถานะ "eligibility" ของ Operator ที่เรียกใช้ Kernel นี้ ในการใช้ Tensor Cores
    - **คำอธิบาย**: บอกว่า Operator ที่เป็นผู้เรียก Kernel นี้ มีคุณสมบัติเหมาะสมที่จะใช้ Tensor Cores หรือไม่
    - **ทำไมต้องกำหนด**: เพื่อช่วยในการทำความเข้าใจว่าถึงแม้ Kernel อาจจะไม่ได้ใช้ Tensor Cores แต่ Operator ต้นทางนั้นมีสิทธิ์ใช้หรือไม่ ซึ่งอาจช่วยในการดีบักปัญหา performance
    - **การใช้งาน**: แสดงในตาราง Kernel เพื่อให้ข้อมูลบริบทเกี่ยวกับ Operator ที่เกี่ยวข้อง

สรุปแล้ว ไฟล์ `consts.py` นี้มีความสำคัญอย่างยิ่งในการทำให้โค้ดของ `torch_tb_profiler` มีความเป็นระเบียบ ง่ายต่อการบำรุงรักษา และช่วยให้ผู้ใช้เข้าใจเมตริกต่างๆ ที่แสดงผลได้ดียิ่งขึ้นผ่านทาง tooltips ที่ให้ข้อมูลอย่างละเอียด
