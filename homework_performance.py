import torch
import time

# -----------------------------------------------------------------------------
# 3.1 Подготовка данных
# -----------------------------------------------------------------------------

def create_matrices():
    """
    Создаёт набор матриц заданных размеров.
    Возвращает словарь {имя: (тензор_cpu, тензор_gpu)}.
    """
    shapes = {
        "64x1024x1024": (64, 1024, 1024),
        "128x512x512": (128, 512, 512),
        "256x256x256": (256, 256, 256),
    }
    matrices = {}
    for name, shape in shapes.items():
        cpu_tensor = torch.rand(shape, dtype=torch.float32)
        if torch.cuda.is_available():
            gpu_tensor = cpu_tensor.cuda()
        else:
            gpu_tensor = None
        matrices[name] = (cpu_tensor, gpu_tensor)
    return matrices


# -----------------------------------------------------------------------------
# 3.2 Функция измерения времени
# -----------------------------------------------------------------------------

def measure_time_cpu(func, *args, repeats=5):
    """
    Измеряет среднее время выполнения функции на CPU.
    """
    # Прогрев
    for _ in range(2):
        func(*args)
    start = time.perf_counter()
    for _ in range(repeats):
        func(*args)
    end = time.perf_counter()
    return (end - start) / repeats * 1000  # в мс


def measure_time_gpu(func, *args, repeats=5):
    """
    Измеряет среднее время выполнения функции на GPU.
    """
    if not torch.cuda.is_available():
        return None
    # Прогрев
    for _ in range(2):
        func(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        func(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats 


# -----------------------------------------------------------------------------
# 3.3 Сравнение операций
# -----------------------------------------------------------------------------

def benchmark_operations():
    """Сравнивает время выполнения операций на CPU и GPU и выводит таблицу."""
    print("3.3 Сравнение операций")
    header = f"{'Операция':25s} | {'CPU (мс)':>9s} | {'GPU (мс)':>9s} | {'Ускорение':>10s}"

    matrices = create_matrices()

    operations = [
        ("Матричное умножение", lambda a, b: torch.matmul(a, b), True),
        ("Поэлементное сложение", lambda a, b: a + b, True),
        ("Поэлементное умножение", lambda a, b: a * b, True),
        ("Транспонирование", lambda a, _: a.permute(*reversed(range(a.ndim))), False),
        ("Сумма всех элементов", lambda a, _: torch.sum(a), False),
    ]

    for size_name, (cpu_tensor, gpu_tensor) in matrices.items():
        print(f"\nРазмер: {size_name}")
        print(header)
        print("-" * len(header))
        for op_name, op_func, needs_second in operations:
            # CPU
            if needs_second:
                cpu_arg2 = torch.rand_like(cpu_tensor)
            else:
                cpu_arg2 = torch.zeros_like(cpu_tensor)  # или любой "заглушечный" тензор

            def cpu_wrapper():
                return op_func(cpu_tensor, cpu_arg2)

            cpu_time = measure_time_cpu(cpu_wrapper)

            # GPU
            if gpu_tensor is not None:
                if needs_second:
                    gpu_arg2 = torch.rand_like(gpu_tensor)
                else:
                    gpu_arg2 = torch.zeros_like(gpu_tensor)

                def gpu_wrapper():
                    return op_func(gpu_tensor, gpu_arg2)

                gpu_time = measure_time_gpu(gpu_wrapper)
                if gpu_time is not None and cpu_time > 0:
                    speedup = cpu_time / gpu_time
                else:
                    speedup = float('inf')
            else:
                gpu_time = None
                speedup = None

            if gpu_time is not None:
                print(f"{op_name:25s} | {cpu_time:9.4f} | {gpu_time:9.4f} | {speedup:10.4f}x")
            else:
                print(f"{op_name:25s} | {cpu_time:9.4f} | {'N/A':9s} | {'N/A':10s}")


benchmark_operations()