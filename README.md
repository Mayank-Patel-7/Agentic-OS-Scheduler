🚀 Agentic OS Scheduler
Dynamic Weight Adaptation for Intelligent CPU Scheduling

⚡ A next-generation CPU scheduler using Machine Learning + Reinforcement Learning to dynamically optimize system performance in real time.

✨ Overview

Traditional schedulers rely on fixed priorities, which fail under changing workloads.
This project introduces an adaptive, learning-based scheduler that continuously improves:

⚡ Latency

⚖️ Fairness

🚀 Throughput

🧠 Core Approach
Component	Role
📊 Gaussian Naïve Bayes	Detects workload (Interactive / Batch / Mixed)
🎯 UCB Bandit	Selects optimal scheduling weights
🔁 Feedback Loop	Learns from system performance
🏗️ System Architecture
Telemetry → Classification → Decision → Weight Update → Scheduling → Feedback

Kernel (Data Plane): Fast scheduling using sched_ext

User Space (Control Plane): Learning and optimization

⚙️ Scheduling Model
score(p) = Σ wi · fi(p)

wi → Adaptive weights

fi(p) → Task features

✔ Weights are updated dynamically based on workload behavior

📈 Performance

⚡ Up to 8.7% lower P99 latency

🎮 Improved interactive responsiveness

⚖️ Higher fairness index

🔁 ~200 ms adaptation speed

🆚 Comparison
Scheduler	Limitation
FCFS	High latency
SJF	Starvation
Round Robin	Inefficient
Priority	Static
Agentic Scheduler	✅ Adaptive & intelligent
🔋 Efficiency

🔻 ~9.9% power reduction

⚡ Smart P-core / E-core utilization

🛠️ Tech Stack

🐧 Linux sched_ext (eBPF)

🧠 Machine Learning (GNB)

🎯 Reinforcement Learning (UCB)

👨‍💻 Authors

Mayank Patel

Rutvik Tayde

⭐ Conclusion

Moving towards self-optimizing operating systems 🚀
