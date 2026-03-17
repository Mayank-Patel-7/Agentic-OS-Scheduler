🚀 Agentic OS Scheduler
Dynamic Weight Adaptation for Smart CPU Scheduling

⚡ An adaptive CPU scheduler that uses ML + Reinforcement Learning to optimize latency, fairness, and throughput in real time.

✨ Highlights

⚡ Up to 8.7% lower P99 latency

🎯 Better interactive responsiveness

⚖️ Improved fairness (Jain’s Index)

🔁 Adapts in ~200 ms to workload changes

🧠 How It Works

📊 Gaussian Naïve Bayes → Detects workload type

🎯 UCB Bandit → Tunes scheduler weights

🔁 Closed-loop system → Continuously learns

🏗️ Architecture

Kernel (Data Plane) → Fast scheduling via sched_ext

User Space (Control Plane) → ML + decision making

📌 Uses telemetry → classify → optimize → update → repeat

⚙️ Core Idea
𝑠
𝑐
𝑜
𝑟
𝑒
(
𝑝
)
=
Σ
𝑤
𝑖
⋅
𝑓
𝑖
(
𝑝
)
score(p)=Σwi⋅fi(p)

Dynamic weights adapt based on system behavior in real time.

🆚 Why Not Traditional Schedulers?
Scheduler	Problem
FCFS	High latency
SJF	Unfair
RR	Inefficient
Priority	Static
✅ Agentic	Adaptive & intelligent
🔋 Efficiency

🔻 ~9.9% power reduction

⚡ Smart P-core / E-core utilization

🛠️ Tech Stack

Linux sched_ext (eBPF)

Machine Learning (GNB)

Reinforcement Learning (UCB)

👨‍💻 Authors

Mayank Patel

Rutvik Tayde

⭐ Final Note

A step toward intelligent, self-optimizing operating systems.
