# ── Import Libraries ─────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.parser import parse
import warnings
warnings.filterwarnings('ignore')

# Global plot style
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 100, 'figure.facecolor': 'white'})

df = pd.read_csv('/content/netflix.csv')
df.head()
