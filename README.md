<h1 align="center">🚀 Agentic OS Scheduler</h1>
<h3 align="center">Dynamic Weight Adaptation for Intelligent CPU Scheduling</h3>

<p align="center">
⚡ A next-generation CPU scheduler using <b>Machine Learning + Reinforcement Learning</b> 
to dynamically optimize system performance in real time.
</p>

<hr>

<h2>✨ Overview</h2>
<p>
Traditional schedulers rely on <b>fixed priorities</b>, which fail under changing workloads.
This project introduces an <b>adaptive, learning-based scheduler</b> that continuously improves:
</p>

<ul>
  <li>⚡ Latency</li>
  <li>⚖️ Fairness</li>
  <li>🚀 Throughput</li>
</ul>

<hr>

<h2>🧠 Core Approach</h2>

<table>
<tr>
<th>Component</th>
<th>Role</th>
</tr>

<tr>
<td>📊 Gaussian Naïve Bayes</td>
<td>Detects workload (Interactive / Batch / Mixed)</td>
</tr>

<tr>
<td>🎯 UCB Bandit</td>
<td>Selects optimal scheduling weights</td>
</tr>

<tr>
<td>🔁 Feedback Loop</td>
<td>Learns from system performance</td>
</tr>
</table>

<hr>

<h2>🏗️ System Architecture</h2>

<pre>
Telemetry → Classification → Decision → Weight Update → Scheduling → Feedback
</pre>

<p>
<b>Kernel (Data Plane):</b> Fast scheduling using <code>sched_ext</code><br>
<b>User Space (Control Plane):</b> Learning and optimization
</p>

<hr>

<h2>⚙️ Scheduling Model</h2>

<pre>
score(p) = Σ wi · fi(p)
</pre>

<ul>
  <li><b>wi</b> → Adaptive weights</li>
  <li><b>fi(p)</b> → Task features</li>
</ul>

<p>✔ Weights update dynamically based on workload behavior</p>

<hr>

<h2>📈 Performance</h2>

<ul>
  <li>⚡ Up to <b>8.7% lower P99 latency</b></li>
  <li>🎮 Improved interactive responsiveness</li>
  <li>⚖️ Higher fairness index</li>
  <li>🔁 ~200 ms adaptation speed</li>
</ul>

<hr>

<h2>🆚 Comparison</h2>

<table>
<tr>
<th>Scheduler</th>
<th>Limitation</th>
</tr>

<tr>
<td>FCFS</td>
<td>High latency</td>
</tr>

<tr>
<td>SJF</td>
<td>Starvation</td>
</tr>

<tr>
<td>Round Robin</td>
<td>Inefficient</td>
</tr>

<tr>
<td>Priority</td>
<td>Static</td>
</tr>

<tr>
<td><b>Agentic Scheduler</b></td>
<td>✅ Adaptive & intelligent</td>
</tr>
</table>

<hr>

<h2>🔋 Efficiency</h2>

<ul>
  <li>🔻 ~9.9% power reduction</li>
  <li>⚡ Smart P-core / E-core utilization</li>
</ul>

<hr>

<h2>🛠️ Tech Stack</h2>

<ul>
  <li>🐧 Linux <code>sched_ext</code> (eBPF)</li>
  <li>🧠 Machine Learning (GNB)</li>
  <li>🎯 Reinforcement Learning (UCB)</li>
  <li>🐍 Currentlly Simulation by Python</li>
</ul>

<hr>

<h2>👨‍💻 Authors</h2>

<ul>
  <li><b>Mayank Patel</b></li>
  <li><b>Rutvik Tayde</b></li>
  <li><b>Kushal Jain</b></li>
</ul>

<hr>

<h2>⭐ Conclusion</h2>

<p align="center">
<b>Moving towards self-optimizing operating systems 🚀</b>
</p>
