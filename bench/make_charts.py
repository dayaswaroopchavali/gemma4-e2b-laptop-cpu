# Benchmark charts (matplotlib): decode tok/s and prefill seconds for stock vLLM, Ollama and the CPU engine on 8 prompts. The numbers are typed in below from docs/RESULTS.md.
import matplotlib; matplotlib.use('Agg')
import pathlib; OUT = pathlib.Path(__file__).resolve().parent.parent / 'charts'
import matplotlib.pyplot as plt, numpy as np
C = {'Stock vLLM': '#2a78d6', 'Ollama': '#eb6834', 'CPU engine': '#1baf7a'}; INK, MUTED, GRID = '#1b1b1a', '#6b6a64', '#e6e5df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': INK, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
# name, prompt tokens, decode tok/s [stock, ollama, engine], prefill s [stock, ollama, engine]
D = [('Repeated\ntext', 1796, [21.07, 24.01, 59.98], [31.3, 17.09, 12.33]), ('Long\ndocument', 6841, [9.94, 17.54, 24.41], [139.0, 73.89, 58.41]),
     ('Message\nsummary', 165, [12.20, 22.42, 29.94], [3.0, 1.43, 1.15]), ('Alert\ntext', 57, [16.83, 22.42, 37.45], [1.25, 0.51, 0.41]),
     ('Call\nrequest', 37, [14.63, 29.24, 43.77], [0.90, 0.36, 0.28]), ('Short\ncommand', 18, [12.61, 19.79, 30.96], [0.50, 0.23, 0.14]),
     ('Tool-call\nJSON', 71, [19.90, 29.23, 38.21], [1.50, 0.65, 0.51]), ('Running\nsummary', 581, [11.19, 24.18, 28.46], [10.35, 5.27, 4.50])]
SYS = list(C)

def grouped(ax, rows, idx, fmt, ymax):
    x = np.arange(len(rows)); w = 0.26
    for i, s in enumerate(SYS):
        v = [r[idx][i] for r in rows]; b = ax.bar(x + (i - 1) * (w + 0.02), v, w, color=C[s], label=s, zorder=3)
        for xi, vi in zip(x + (i - 1) * (w + 0.02), v): ax.text(xi, vi + ymax * 0.012, fmt(vi), ha='center', va='bottom', fontsize=7.6, color=INK, bbox=dict(fc='white', ec='none', pad=0.8), zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([f'{r[0]}\n{r[1]:,} tok' for r in rows], fontsize=8.6); ax.set_ylim(0, ymax)
    ax.grid(axis='y', color=GRID, zorder=0); ax.tick_params(axis='x', length=0, pad=6); ax.tick_params(axis='y', length=0)

# 1. decode
fig, ax = plt.subplots(figsize=(12, 5.6)); grouped(ax, D, 2, lambda v: f'{v:.1f}', 68)
ax.set_ylabel('Decode speed (tokens per second)')
m = [np.mean([r[2][i] for r in D]) for i in range(3)]
fig.text(0.07, 0.965, f'Higher is better. Means over 8 prompts: stock vLLM {m[0]:.1f}, Ollama {m[1]:.1f}, CPU engine {m[2]:.1f} tok/s. Ryzen 7 7730U, 8 cores, one stream.', fontsize=9.6, color=MUTED, ha='left', va='top')
ax.legend(loc='upper right', bbox_to_anchor=(1, 0.995), frameon=False, ncol=3, fontsize=9.5)
fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.17); fig.savefig(OUT / 'decode_tok_s.png', dpi=200); plt.close(fig)

# 2. prefill (two scales)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={'width_ratios': [2, 6.2], 'wspace': 0.1})
grouped(a1, D[:2], 3, lambda v: f'{v:.0f}' if v >= 100 else f'{v:.1f}', 160); grouped(a2, D[2:], 3, lambda v: f'{v:.2f}' if v < 10 else f'{v:.1f}', 12)
a1.set_ylabel('Seconds to first token (lower is better)'); a1.set_title('Long prompts', loc='left', fontsize=10, color=MUTED, pad=8); a2.set_title('Short prompts (different scale)', loc='left', fontsize=10, color=MUTED, pad=8)
fig.text(0.07, 0.975, 'Cold prompt to first token, seconds. Stock vLLM time includes one decode step. Engine prefill uses the bounded int8 path.', fontsize=9.6, color=MUTED, ha='left', va='top')
h, l = a2.get_legend_handles_labels(); a2.legend(h, l, loc='upper right', frameon=False, ncol=3, fontsize=9.5)
fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.17); fig.savefig(OUT / 'prefill_seconds.png', dpi=200); plt.close(fig)
print('ok')
