# 🤖 Generative AI-Powered Data Analytics Platform

A comprehensive, modular AI-driven analytics platform that combines advanced machine learning, blockchain technology, federated learning, and natural language processing to provide end-to-end data analytics solutions.

## 🌟 Features

### Core Analytics Modules
- **📊 Data Upload & Management** - Support for CSV, Excel, and various data formats
- **🧹 Data Cleaning & Preprocessing** - Automated data quality improvements and feature engineering
- **📈 Advanced Analytics** - Statistical analysis, correlation studies, and insights generation
- **🤖 AI Data Generation** - Synthetic data creation using advanced generative models
- **💡 AI-Powered Insights** - Machine learning-driven recommendations and pattern detection

### Advanced AI & ML
- **🧠 Advanced ML Models & AutoML** - 20+ algorithms with automated hyperparameter optimization
- **🏗️ Ensemble Methods** - Voting, bagging, boosting, and stacking classifiers
- **🧠 Deep Learning Simulation** - Neural network architectures and training
- **📊 Model Comparison & Analysis** - Comprehensive model evaluation and benchmarking

### Cutting-Edge Technologies
- **🔗 Blockchain Manager** - Data integrity, audit trails, and smart contracts
- **🤝 Federated Learning** - Multi-party machine learning with privacy preservation
- **🗣️ Natural Language Query** - Chat with your data using conversational AI
- **🎮 Scenario Simulation** - What-if analysis and predictive modeling
- **⚡ System Optimizer** - Performance optimization and resource management

### Interactive Dashboards
- **📊 Dynamic Dashboard** - Real-time visualizations and interactive charts
- **📋 System Logs** - Comprehensive logging and monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shaikhamzasaifuddin402/Ai-data-analyst.git
   cd Ai-data-analyst
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the platform**
   Open your browser and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Getting Started
1. **Upload Data**: Start by uploading your dataset in the "📤 Data Upload" tab
2. **Clean Data**: Use the "🧹 Data Cleaning" tab to preprocess your data
3. **Explore Analytics**: Navigate through various analysis tabs based on your needs
4. **Generate Insights**: Use AI-powered features for automated insights and recommendations

### Key Workflows

#### Basic Analytics Workflow
```
Data Upload → Data Cleaning → Advanced Analytics → AI Insights → Dynamic Dashboard
```

#### Advanced ML Workflow
```
Data Upload → Data Cleaning → Advanced ML Models → Model Comparison → Results Export
```

#### Blockchain Audit Workflow
```
Data Upload → Blockchain Registration → Smart Contract Deployment → Audit Trail Review
```

## 🏗️ Architecture

### System Components
- **Frontend**: Streamlit-based web interface
- **Backend**: Modular Python agents for specialized tasks
- **AI Core**: Generative analytics engine with multiple ML models
- **Blockchain**: Distributed ledger for data integrity
- **Storage**: Session-based data management with optional persistence

### Module Structure
```
├── app.py                 # Main application entry point
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── agents/              # Core functionality modules
│   ├── data_processor.py
│   ├── ai_core.py
│   ├── blockchain_manager.py
│   ├── federated_learning.py
│   ├── advanced_ml_models.py
│   └── ...
└── utils/               # Utility functions
    └── data_generator.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Advanced Configuration
Modify `config.py` to customize:
- Model parameters
- Performance settings
- Security configurations
- Feature toggles

## 🧩 Optional NLP Dependencies
- The app now guards optional NLP imports. If `transformers` or TensorFlow is not available or incompatible, NLP features are gracefully disabled and the rest of the app continues to run.
- If you want Transformer pipelines enabled with TensorFlow, and you encounter the Keras 3 compatibility error, you can:
  - Install `tf-keras` (`pip install tf-keras`), or
  - Prefer PyTorch backend (`pip install torch`) to avoid TensorFlow/Keras coupling, or
  - Disable TensorFlow in Transformers by setting `TRANSFORMERS_NO_TF=1` in your environment.

## 📊 Supported Data Formats

- **CSV Files** (.csv)
- **Excel Files** (.xlsx, .xls)
- **JSON Files** (.json)
- **Parquet Files** (.parquet)
- **Data Connectors** (optional; add SQL/MongoDB if needed)

## 🤖 AI Models & Algorithms

### Machine Learning Algorithms
- **Classification**: Random Forest, SVM, Gradient Boosting, Neural Networks
- **Regression**: Linear/Polynomial Regression, Ridge, Lasso, ElasticNet
- **Clustering**: K-Means, DBSCAN, Hierarchical Clustering
- **Ensemble Methods**: Voting, Bagging, Boosting, Stacking

### Deep Learning Models
- **Neural Networks**: Feedforward, CNN, RNN, LSTM
- **Generative Models**: GANs, VAEs, Transformers
- **AutoML**: Automated model selection and hyperparameter tuning

## 🔐 Security & Privacy

### Blockchain Features
- **Data Integrity**: Cryptographic hashing and verification
- **Audit Trails**: Immutable transaction records
- **Smart Contracts**: Automated data validation rules

### Privacy Protection
- **Federated Learning**: Decentralized model training
- **Differential Privacy**: Statistical privacy guarantees
- **Secure Multi-party Computation**: Privacy-preserving analytics

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=agents tests/
```

### Test Coverage
- Unit tests for all core modules
- Integration tests for workflows
- Performance benchmarks
- Security validation tests

## 📈 Performance

### Optimization Features
- **Caching**: Streamlit caching for improved performance
- **Lazy Loading**: On-demand module loading
- **Memory Management**: Efficient data handling
- **Parallel Processing**: Multi-threaded computations

### Scalability
- Supports datasets up to 10M+ rows
- Distributed computing capabilities
- Cloud deployment ready
- Edge computing support

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes
5. Run tests: `pytest`
6. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues
- **Import Errors**: Ensure all dependencies are installed
- **Memory Issues**: Reduce dataset size or increase system RAM
- **Performance**: Enable caching and optimize data types

### Getting Help
- Check the [Issues](issues) page for known problems
- Review the documentation for detailed usage instructions
- Contact the development team for technical support

## 🔮 Future Roadmap

### Planned Features
- **Real-time Data Streaming** - Live data processing capabilities
- **Advanced Visualization** - 3D plots and interactive dashboards
- **API Integration** - RESTful API for external integrations
- **Mobile Support** - Responsive design for mobile devices
- **Multi-language Support** - Internationalization features

### Research Areas
- **Quantum Computing Integration** - Quantum-enhanced algorithms
- **Explainable AI** - Enhanced model interpretability
- **Edge AI** - Lightweight models for edge deployment
- **Automated Feature Engineering** - AI-driven feature creation

---

**Built with ❤️ using Python, Streamlit, and cutting-edge AI technologies**

For more information, visit our [documentation](docs/) or contact the development team.