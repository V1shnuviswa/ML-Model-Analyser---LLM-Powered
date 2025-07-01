# 🔍 ML-Model-Analyser---LLM-Powered

## 🚀 Overview
**AI Model Analyzer** is an advanced platform designed to evaluate and optimize machine learning models by assessing both their performance metrics and energy efficiency.

As AI powers critical applications across industries, optimizing models for computational cost and sustainability has become essential.  
This tool automates the evaluation process by analyzing Python-based ML scripts, offering insights into:

- 📈 Accuracy  
- ⏱ Execution time  
- 🖥️ Resource usage  
- ⚡ Energy consumption  

It also recommends alternative models that improve performance while reducing energy demands.  
The platform generates comprehensive reports with data visualizations, enabling AI practitioners to make data-driven decisions and build models that balance power, efficiency, and sustainability.

---

## 🌟 Key Features

- **Efficient Model Evaluation**  
  Systematically assess ML models for accuracy, execution speed, and energy consumption.

- **Energy Optimization**  
  Analyze and minimize computational costs to support environmentally responsible AI development.

- **Sustainable AI Development**  
  Promote green AI by reducing carbon footprints through energy-efficient modeling.

- **Automation & Scalability**  
  Automated, scalable analysis system suitable for practitioners across domains.

- **Data-Driven Insights**  
  Generate detailed reports with visualizations for informed decision-making.

- **AI Assistant**  
  NLP-based assistant (powered by **LLaMA 2**) offers explanations, suggests optimizations, and makes the platform accessible for all expertise levels.

- **Community Collaboration**  
  Encourages sharing best practices and sustainable AI strategies.

---

## 🧩 System Architecture
![image alt](https://github.com/V1shnuviswa/ML-Model-Analyser-LLM-Powered/blob/6a45922a82faefb87d36c251d00fd17fd1505c57/ML%20Model%20Analyzer.png)
| Component              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| 📚 Model Analysis      | Structured approach to assess efficiency, accuracy, and computational costs |
| 🏗 Data Processing      | Uses NumPy & Pandas for high-speed numerical and tabular data handling      |
| 🎯 Model Evaluation     | Scikit-learn evaluates metrics like accuracy, precision, recall, F1-score    |
| 📈 Visualization        | Matplotlib & Seaborn generate clear, interactive data visualizations         |
| 🤖 AI Assistant         | LLaMA 2 NLP analyzes model architectures & suggests improvements             |
| ⚡ Optimization         | Hyperparameter tuning via Grid & Random Search + parallel processing         |
| 📝 Reporting            | PyPDF generates structured, professional PDF reports                         |
| 🔄 Deployment/Maintenance | Continuous improvements via updates and user feedback integration          |

---

## 📂 Datasets Used

- **Feature Engineering Data**  
  Preprocessed with NumPy & Pandas for optimal performance.

- **Model Performance Data**  
  Metrics from Scikit-learn evaluations (accuracy, precision, recall, F1-score).

- **Visualization Data**  
  Structured datasets for plotting trends with Matplotlib & Seaborn.

- **AI Assistant Data**  
  Model architecture details for NLP-driven insights.

---

## 🛠 Tools & Technologies

| Technology     | Purpose                                           |
|----------------|---------------------------------------------------|
| React.js       | Frontend UI: interactive, responsive components   |
| Flask          | Python backend API & business logic               |
| Scikit-learn   | ML model evaluation, comparison, tuning           |
| NumPy          | High-speed numerical computations                 |
| Pandas         | Data manipulation & analysis                      |
| Matplotlib     | Plot generation                                   |
| Seaborn        | Statistical visualization                         |
| PyPDF          | PDF report generation                             |
| LLaMA 2        | AI assistant for insights & suggestions           |
| Grid/Random Search | Hyperparameter tuning                        |

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.8+
- Node.js (for React frontend)
- Virtual environment (recommended)

### 🔧 Setup Instructions (Backend)

```bash
# Clone the repository
git clone https://github.com/your-username/ML-Model-Analyser---LLM-Powered.git
cd ML-Model-Analyser---LLM-Powered/backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py
