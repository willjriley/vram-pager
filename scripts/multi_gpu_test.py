"""
Multi-GPU Testing Script — Spin up RunPod instances, compile kernel, benchmark, teardown.

Tests across multiple GPU architectures to verify compatibility and gather benchmarks.
Automatically destroys pods when done.
"""
import json
import urllib.request
import time
import sys
import os

API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# GPU configs to test
GPU_CONFIGS = [
    {"name": "vram-pager-ampere86", "gpu": "NVIDIA RTX A6000", "arch": "sm_86", "desc": "Ampere (RTX 3090/A6000)"},
    {"name": "vram-pager-ada89", "gpu": "NVIDIA L40S", "arch": "sm_89", "desc": "Ada Lovelace (RTX 4090/L40S)"},
]

KERNEL_SOURCE = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void dequant_int8(
    const int8_t* quantized, const float* scales, float* output,
    int n, int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = (float)quantized[i] * scales[i / block_size];
    }
}

extern "C" __declspec(dllexport) void launch_dequant_int8(
    const int8_t* q, const float* s, float* o, int n, int bs, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int8<<<blocks, threads, 0, stream>>>(q, s, o, n, bs);
}
'''

BENCHMARK_CODE = '''
import torch
import time
import subprocess
import os
import json

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CUDA: {torch.version.cuda}")

# Write kernel source
os.makedirs("/workspace/vram_pager", exist_ok=True)
with open("/workspace/vram_pager/dequant.cu", "w") as f:
    f.write(KERNEL_SRC)

# Compile
ARCH = "ARCH_PLACEHOLDER"
print(f"\\nCompiling for {ARCH}...")
t0 = time.time()
result = subprocess.run([
    "nvcc", "-O2", "--shared",
    f"-gencode=arch={ARCH.replace('sm_','compute_')},code={ARCH}",
    "-o", "/workspace/vram_pager/dequant.so",
    "/workspace/vram_pager/dequant.cu",
    "-lcudart"
], capture_output=True, text=True)
compile_time = time.time() - t0

if result.returncode != 0:
    print(f"COMPILE FAILED: {result.stderr[:500]}")
else:
    print(f"Compiled in {compile_time:.1f}s")

    # Load and test
    import ctypes
    dll = ctypes.CDLL("/workspace/vram_pager/dequant.so")
    dll.launch_dequant_int8.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]

    N = 26_000_000
    BLOCK = 128

    original = torch.randn(N, device="cuda")
    scales = original.reshape(-1, BLOCK).abs().max(dim=1).values / 127.0
    quantized = (original.reshape(-1, BLOCK) / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8).reshape(-1)
    output = torch.empty(N, dtype=torch.float32, device="cuda")

    # Warmup
    for _ in range(10):
        dll.launch_dequant_int8(quantized.data_ptr(), scales.data_ptr(), output.data_ptr(), N, BLOCK, None)
    torch.cuda.synchronize()

    # Benchmark kernel
    t0 = time.time()
    for _ in range(1000):
        dll.launch_dequant_int8(quantized.data_ptr(), scales.data_ptr(), output.data_ptr(), N, BLOCK, None)
    torch.cuda.synchronize()
    kernel_ms = (time.time() - t0) / 1000 * 1000

    # Correctness
    pt_out = (quantized.float().reshape(-1, BLOCK) * scales.unsqueeze(1)).reshape(-1)
    max_diff = (output - pt_out).abs().max().item()

    # PCIe bandwidth
    fp32_cpu = torch.randn(N, device="cpu").pin_memory()
    int8_cpu = torch.randint(-128, 127, (N,), dtype=torch.int8, device="cpu").pin_memory()
    scales_cpu = torch.randn(N // BLOCK, device="cpu").pin_memory()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        _ = fp32_cpu.cuda(non_blocking=True)
        torch.cuda.synchronize()
    fp32_ms = (time.time() - t0) / 50 * 1000

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        gi = int8_cpu.cuda(non_blocking=True)
        gs = scales_cpu.cuda(non_blocking=True)
        torch.cuda.synchronize()
        dll.launch_dequant_int8(gi.data_ptr(), gs.data_ptr(), output.data_ptr(), N, BLOCK, None)
        torch.cuda.synchronize()
        del gi, gs
    int8_ms = (time.time() - t0) / 50 * 1000

    results = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "cuda_version": torch.version.cuda,
        "arch": ARCH,
        "compile_time_s": round(compile_time, 1),
        "kernel_ms": round(kernel_ms, 3),
        "max_error": max_diff,
        "correct": max_diff < 1e-5,
        "fp32_transfer_ms": round(fp32_ms, 1),
        "int8_transfer_decompress_ms": round(int8_ms, 1),
        "net_speedup": round(fp32_ms / int8_ms, 2),
        "fp32_bandwidth_gbs": round(N * 4 / 1e6 / fp32_ms * 1000, 0),
    }

    print(f"\\n=== RESULTS ===")
    print(json.dumps(results, indent=2))
'''


def graphql(query):
    import subprocess
    result = subprocess.run([
        "curl", "-s", "-H", "Content-Type: application/json",
        "-d", json.dumps({"query": query}),
        f"https://api.runpod.io/graphql?api_key={API_KEY}"
    ], capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def create_pod(name, gpu_type):
    query = 'mutation { podFindAndDeployOnDemand(input: { name: "' + name + '", gpuTypeId: "' + gpu_type + '", gpuCount: 1, volumeInGb: 10, containerDiskInGb: 20, imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", volumeMountPath: "/workspace", minVcpuCount: 4, minMemoryInGb: 16, ports: "8888/http" }) { id name machine { gpuDisplayName } } }'
    result = graphql(query)
    if "errors" in result:
        return None, result["errors"][0]["message"]
    pod = result["data"]["podFindAndDeployOnDemand"]
    return pod["id"], pod["machine"]["gpuDisplayName"]


def wait_for_pod(pod_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        query = '{ pod(input: {podId: "' + pod_id + '"}) { runtime { uptimeInSeconds } } }'
        result = graphql(query)
        rt = result["data"]["pod"].get("runtime")
        if rt and rt.get("uptimeInSeconds", 0) > 0:
            return True
        time.sleep(15)
    return False


def destroy_pod(pod_id):
    graphql('mutation { podTerminate(input: {podId: "' + pod_id + '"}) }')


def run_on_pod(pod_id, code, timeout=300):
    """Execute code on pod via Jupyter WebSocket API."""
    import websocket

    jupyter_url = f"https://{pod_id}-8888.proxy.runpod.net"

    # Wait for Jupyter
    for _ in range(20):
        try:
            req = urllib.request.urlopen(f"{jupyter_url}/api?token=dev", timeout=5)
            if req.status == 200:
                break
        except:
            pass
        time.sleep(10)

    import requests
    session = requests.Session()
    session.headers.update({"Authorization": "token dev"})

    kernels = session.get(f"{jupyter_url}/api/kernels?token=dev").json()
    if isinstance(kernels, list) and len(kernels) > 0:
        kid = kernels[0]["id"]
    else:
        kid = session.post(f"{jupyter_url}/api/kernels?token=dev").json()["id"]

    ws_url = jupyter_url.replace("https://", "wss://") + f"/api/kernels/{kid}/channels?token=dev"
    ws = websocket.create_connection(ws_url, timeout=30)

    mid = f"run_{time.time()}"
    msg = {
        "header": {"msg_id": mid, "msg_type": "execute_request", "username": "", "session": "", "date": "", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True, "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "buffers": [], "channel": "shell"
    }
    ws.send(json.dumps(msg))

    output = ""
    start = time.time()
    while time.time() - start < timeout:
        try:
            ws.settimeout(15)
            r = json.loads(ws.recv())
            if r.get("parent_header", {}).get("msg_id") != mid:
                continue
            if r.get("msg_type") == "stream":
                output += r["content"]["text"]
                print(r["content"]["text"], end="", flush=True)
            elif r.get("msg_type") == "error":
                output += f"ERROR: {r['content'].get('evalue', '')}\n"
                print(f"ERROR: {r['content'].get('evalue', '')}")
            elif r.get("msg_type") == "execute_reply":
                break
        except:
            continue
    ws.close()
    return output


def main():
    all_results = []

    for config in GPU_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Testing: {config['desc']} ({config['gpu']})")
        print(f"{'='*60}")

        # Create pod
        print(f"Creating pod...")
        pod_id, gpu_name = create_pod(config["name"], config["gpu"])
        if pod_id is None:
            print(f"  FAILED: {gpu_name}")
            continue
        print(f"  Pod: {pod_id} ({gpu_name})")

        try:
            # Wait for pod
            print(f"  Waiting for pod to start...")
            if not wait_for_pod(pod_id):
                print(f"  TIMEOUT waiting for pod")
                continue

            # Install deps
            print(f"  Installing ninja...")
            run_on_pod(pod_id, "import subprocess; subprocess.run(['pip', 'install', 'ninja', 'websocket-client'], capture_output=True); print('deps OK')", timeout=60)

            # Run benchmark
            print(f"  Running benchmark...")
            code = BENCHMARK_CODE.replace("ARCH_PLACEHOLDER", config["arch"])
            code = code.replace("KERNEL_SRC", repr(KERNEL_SOURCE))
            # Fix: need to define KERNEL_SRC variable
            full_code = f'KERNEL_SRC = {repr(KERNEL_SOURCE)}\n' + code.replace("KERNEL_SRC", "KERNEL_SRC")

            # Actually just inline it properly
            test_code = f'''
import torch, time, subprocess, os, json, ctypes

print(f"GPU: {{torch.cuda.get_device_name(0)}}")
print(f"VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")

os.makedirs("/workspace/vram_pager", exist_ok=True)
with open("/workspace/vram_pager/dequant.cu", "w") as f:
    f.write("""{KERNEL_SOURCE.replace('"', chr(92)+'"')}""")

ARCH = "{config['arch']}"
print(f"Compiling for {{ARCH}}...")
t0 = time.time()
r = subprocess.run(["nvcc", "-O2", "--shared",
    f"-gencode=arch={{ARCH.replace('sm_','compute_')}},code={{ARCH}}",
    "-o", "/workspace/vram_pager/dequant.so",
    "/workspace/vram_pager/dequant.cu", "-lcudart"],
    capture_output=True, text=True)
ct = time.time() - t0
if r.returncode != 0:
    print(f"FAILED: {{r.stderr[:300]}}")
else:
    print(f"Compiled in {{ct:.1f}}s")
    dll = ctypes.CDLL("/workspace/vram_pager/dequant.so")
    dll.launch_dequant_int8.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]

    N, B = 26000000, 128
    orig = torch.randn(N, device="cuda")
    sc = orig.reshape(-1, B).abs().max(dim=1).values / 127.0
    q = (orig.reshape(-1, B) / sc.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8).reshape(-1)
    out = torch.empty(N, dtype=torch.float32, device="cuda")

    for _ in range(10):
        dll.launch_dequant_int8(q.data_ptr(), sc.data_ptr(), out.data_ptr(), N, B, None)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(1000):
        dll.launch_dequant_int8(q.data_ptr(), sc.data_ptr(), out.data_ptr(), N, B, None)
    torch.cuda.synchronize()
    kms = (time.time()-t0)/1000*1000

    pt = (q.float().reshape(-1, B) * sc.unsqueeze(1)).reshape(-1)
    err = (out - pt).abs().max().item()

    f32 = torch.randn(N, device="cpu").pin_memory()
    i8 = torch.randint(-128, 127, (N,), dtype=torch.int8, device="cpu").pin_memory()
    sc2 = torch.randn(N//B, device="cpu").pin_memory()

    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50):
        _ = f32.cuda(non_blocking=True); torch.cuda.synchronize()
    f32ms = (time.time()-t0)/50*1000

    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50):
        gi = i8.cuda(non_blocking=True); gs = sc2.cuda(non_blocking=True)
        torch.cuda.synchronize()
        dll.launch_dequant_int8(gi.data_ptr(), gs.data_ptr(), out.data_ptr(), N, B, None)
        torch.cuda.synchronize(); del gi, gs
    i8ms = (time.time()-t0)/50*1000

    res = {{"gpu": torch.cuda.get_device_name(0), "vram_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9,1),
        "arch": ARCH, "compile_s": round(ct,1), "kernel_ms": round(kms,3), "error": err, "correct": err<1e-5,
        "fp32_ms": round(f32ms,1), "int8_ms": round(i8ms,1), "speedup": round(f32ms/i8ms,2),
        "bandwidth_gbs": round(N*4/1e6/f32ms*1000,0)}}
    print(f"\\nRESULTS: {{json.dumps(res, indent=2)}}")
'''
            output = run_on_pod(pod_id, test_code, timeout=180)

            # Extract results
            if "RESULTS:" in output:
                results_str = output.split("RESULTS:")[1].strip()
                try:
                    results = json.loads(results_str)
                    all_results.append(results)
                except:
                    pass

        finally:
            # ALWAYS destroy the pod
            print(f"\n  Destroying pod {pod_id}...")
            destroy_pod(pod_id)
            print(f"  Pod destroyed.")

    # Summary
    print(f"\n\n{'='*60}")
    print(f"MULTI-GPU BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n{r['gpu']} ({r['arch']}, {r['vram_gb']}GB VRAM):")
        print(f"  Kernel: {r['kernel_ms']:.3f}ms | Correct: {r['correct']}")
        print(f"  FP32 transfer: {r['fp32_ms']}ms | INT8+decompress: {r['int8_ms']}ms")
        print(f"  Net speedup: {r['speedup']}x | PCIe: {r['bandwidth_gbs']} GB/s")

    # Save results
    with open("os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research")/multi_gpu_benchmarks.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to research/multi_gpu_benchmarks.json")


if __name__ == "__main__":
    main()
