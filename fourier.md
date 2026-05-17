# Fourier Transform

Any signal, periodic or not, can be represented in terms of its frequency components using the Fourier Transform.

# Discrete Fourier Transform (DFT)
The discrete transform of a real signal can be expressed as:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} kn}$$

Where:

- $x[n]$ is the value of the signal at time $n$ (input sample).
- $X[k]$ is the frequency result for the index $k$ (it is a complex number $a + b j$).
- $N$ is the total number of samples.
- $e^{-j \dots}$ is the complex exponential (through the Euler formula can be represented into sin and cos).

In the DFT output, a complex number $a + jb$ is not a simple data point; it is a vector that describes a specific **sine wave**. Every complex number at a certain position $k$ tells you two essential things about that frequency:
- **Amplitude** (How strong is it?): The amplitude is calculated using the magnitude of the complex number: $$\sqrt{a^2 + b^2}$$
It tells you the "weight" or power of that specific frequency within the total signal.
For example, if you see a high magnitude at 50 Hz, it means your signal contains a very strong wave oscillating 50 times per second.

- **Phase** (When does it start?): The phase is determined by the angle of the vector: $$\theta = \arctan(b/a)$$
The phase describes the time shift of the wave. Two waves can have the exact same frequency, but one might start at its peak while the other starts at zero. Without the complex number (and its imaginary part), we would have no way of knowing how these waves are aligned in time.

## What is $X[k]$? 
In the frequency domain, $X[k]$ is the $k$-th element of your FFT output. It is not just a single value, but a complex number that acts as a "report" for one specific frequency.

Think of $X[k]$ as a vector that contains two vital pieces of information about a specific sine wave:

- **Magnitude (The Strength)**: It represents the **Amplitude**. It shows how much of that specific frequency exists in your signal. A high magnitude means that this specific frequency is dominant; a magnitude near zero means that frequency is not present.

- **Phase (The Timing)**: It represents the Time Shift. It tells you where the wave starts at $t=0$. Two waves can have the same frequency and amplitude, but if one is shifted (delayed) compared to the other, they will have different phase values.

$X[k]$ is often called a **Frequency Bin**. The index $k$ tells you which frequency you are looking at.

- **When the magnitude of $X[k]$ is LARGE:** It means that the frequency associated with the index $k$ is a dominant and real component of your original signal. The signal contains a very strong wave oscillating at that exact speed. In other words, a large portion of the signal's total energy is concentrated at this frequency.

- **When the magnitude of $X[k]$ is SMALL (or near zero):** It means that the frequency associated with the index $k$ is essentially absent from your signal. The original signal does not naturally vibrate at that speed. Any tiny value you see is usually just random background noise or minor mathematical interference, not a real wave.

### The relationship between $k$ and $f_s$

The relationship between $k$ (the bin index) and $f_s$ (the sampling frequency) is what allows us to translate abstract math into real-world units (Hertz). Without $f_s$, the index $k$ is just a position in an array. With $f_s$, $k$ becomes a specific frequency.

The most important equation to remember is:
$$f = k \cdot \frac{f_s}{N} = k \cdot \frac{1}{T}$$

Where: 

- $f$: The actual frequency in Hertz (Hz), which I'm looking at.
- $k$: The index in the FFT array ($0, 1, 2, \dots$).
- $f_s$: The sampling frequency (samples per second).
- $N$: The total number of samples you collected.

The term $f_s/N$ is called the **Frequency Resolution**. It represents the "size" of each step (or "bin") in your FFT.  
If $f_s = 1000$ Hz and $N = 1000$, then each step is 1 Hz. $k=1$ is 1 Hz, $k=50$ is 50 Hz.  
If $f_s = 1000$ Hz and $N = 2000$, then each step is 0.5 Hz. $k=1$ is 0.5 Hz, $k=100$ is 50 Hz. 

*Note*: To see more detail in your frequencies (smaller steps), you need to either increase the recording time ($N$) or decrease the sampling rate ($f_s$).

## $f_s$, $N$ and $T$
1. **The Recording Duration ($T$)**: The total amount of time you spend "recording" or observing the signal (e.g., 2 seconds). It dictates your Frequency Resolution ($\Delta f$). The formula is $\Delta f = 1 / T$. If you record for 1 second, your FFT bins are 1 Hz apart. If you record for 10 seconds, your bins are 0.1 Hz apart.  
If you want to zoom in and distinguish very close frequencies (like telling apart 50.0 Hz from 50.1 Hz), you must increase $T$ (record for a longer time).

2. **The Sampling Frequency ($f_s$)**: The "framerate" of your recording. It is how many samples the computer takes every single second (e.g., 1000 Hz means 1000 snapshots per second).  
It dictates your **Maximum Observable Frequency**. It determines the maximum frequency of a wave you can see. It is limited exactly to $f_s / 2$ (The Nyquist Limit).  
Imagine trying to draw a wave. To successfully prove that a wave went up and down (one complete cycle), you need to capture at least two points: the highest peak and the lowest valley. If a wave oscillates 500 times a second (500 Hz), there are 1000 peaks and valleys every second (500 peaks + 500 valleys). Therefore, your camera must "click" at least 1000 times a second ($f_s = 1000$) just to catch the minimum points to draw the wave. If your $f_s$ is slower than that, you miss the peaks, and the computer connects the wrong dots, creating a fake, slower wave. This error is called **Aliasing**.  
Therefore the highest frequency you can possibly measure is always exactly half of your sampling rate ($f_s / 2$). If you want to see a 20 kHz audio wave, you must set $f_s$ to at least 40 kHz.  

*To sample a signal without losing information and without causing distortions (aliasing), your sampling frequency ($f_s$) must be strictly greater than twice the maximum frequency present in the signal ($f_{max}$). As a mathematical formula:* $$f_s > 2 \cdot f_{max}$$

3. **The Number of Samples ($N$)**: The total amount of individual data points you collect.
$$N = f_s \cdot T$$
If you want a huge frequency range (high $f_s$) AND incredible detail (long $T$), $N$ becomes a massive number. Your FFT will require a lot of memory and processing power to calculate.