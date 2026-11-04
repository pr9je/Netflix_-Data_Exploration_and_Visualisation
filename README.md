# Netflix Data Exploration and Visualisation

## Project Overview

This project performs **Exploratory Data Analysis (EDA)** on the Netflix dataset to uncover insights into its content library, including movies, TV shows, genres, countries, directors, actors, ratings, and content release trends.

Using Python and data visualisation libraries, the project explores content distribution, identifies patterns, and analyses Netflix's content catalogue through statistical summaries and visual representations.

## Objectives

- Analyse the distribution of Movies and TV Shows on Netflix.
- Explore content availability across different countries.
- Identify popular genres, actors, and directors.
- Examine content ratings and movie duration patterns.
- Analyse content addition trends over time.
- Compare Netflix content trends across different categories.
- Perform country-specific analysis, particularly for the USA.

## Technologies & Libraries

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualisation:** Matplotlib, Seaborn, Plotly Express
- **Statistical Analysis:** Descriptive Statistics
- **Environment:** Google Colab, Jupyter Notebook

## Project Workflow

1. **Data Loading & Exploration**
   - Load the Netflix dataset.
   - Examine dataset dimensions, columns, and data types.
   - Calculate basic statistical metrics.

2. **Data Cleaning & Preprocessing**
   - Identify and handle missing values.
   - Convert appropriate columns into categorical data types.
   - Process multi-valued columns such as country, cast, director, and genres.
   - Extract and transform date-related features.
   - Prepare separate datasets for Movies and TV Shows.

3. **Exploratory Data Analysis**
   - Perform univariate and bivariate analysis.
   - Analyse categorical and numerical variables.
   - Explore content distribution by genre, country, rating, and duration.
   - Investigate content addition and release-year trends.

4. **Data Visualisation**
   - Generate bar charts, pie charts, histograms, and KDE plots.
   - Create comparative visualisations using Seaborn and Matplotlib.
   - Build interactive trend visualisations using Plotly Express.

5. **Country-Specific Analysis**
   - Analyse USA-based Movies and TV Shows.
   - Explore genre, actor, director, and content release trends.

## Key Areas of Analysis

| Analysis | Description |
|---|---|
| Content Distribution | Movies vs TV Shows |
| Genre Analysis | Most frequently occurring genres |
| Country Analysis | Top content-producing countries |
| Rating Analysis | Distribution of content ratings |
| Duration Analysis | Movie duration and TV Show patterns |
| Actor Analysis | Actors with the highest title counts |
| Director Analysis | Directors with the highest title counts |
| Time-Series Analysis | Content addition and release trends |
| USA Analysis | Country-specific content exploration |

## Repository Structure

```text
Netflix_-Data_Exploration_and_Visualisation/
│
├── images/
│
├── main.py
│
└── README.md
```

## How to Run the Project

**Option 1: Google Colab**

1. Clone or download this repository.
2. Open `main.py` in Google Colab or copy the code into a notebook.
3. Upload the Netflix dataset (`netflix.csv`) to the Colab environment.
4. Install the required libraries if they are not already available.
5. Execute the analysis cells sequentially.

**Option 2: Local Environment**

Clone the repository:

```bash
git clone https://github.com/pr9je/Netflix_-Data_Exploration_and_Visualisation.git
```

Navigate to the project directory:

```bash
cd Netflix_-Data_Exploration_and_Visualisation
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn python-dateutil plotly
```

Place `netflix.csv` in the project directory and update the dataset path in `main.py`:

```python
df = pd.read_csv("netflix.csv")
```

Run the script:

```bash
python main.py
```

## Key Learnings

- Hands-on experience with data cleaning and preprocessing.
- Improved understanding of exploratory data analysis.
- Practical application of descriptive statistics.
- Experience analysing categorical and numerical variables.
- Developed data visualisation and storytelling skills.
- Gained insights into content distribution and temporal trends.

## Author

**Mitesh Prajapati**

GitHub: [@pr9je](https://github.com/pr9je)

---

***This project was developed as part of my Data Science and Machine Learning learning journey.***
