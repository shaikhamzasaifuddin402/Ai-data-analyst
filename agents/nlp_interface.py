"""
Natural Language Interface Module
Implements conversational AI for natural language queries and interactive data exploration
Based on the research paper specifications for conversational interface capabilities
"""

import pandas as pd
import numpy as np
import openai
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
# Optional: guard Transformers import to avoid Keras 3 errors
try:
    from transformers import pipeline  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    pipeline = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False
    logging.getLogger(__name__).warning("Transformers unavailable in NLP interface: %s", e)
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NaturalLanguageInterface:
    """
    Natural Language Interface for conversational data analytics
    Enables users to interact with data using natural language queries
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the Natural Language Interface"""
        self.openai_api_key = openai_api_key
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize query patterns and templates
        self.query_patterns = self._initialize_query_patterns()
        
        # Initialize NLP components
        self.question_answerer = None
        self._initialize_nlp_components()
        
        # Context tracking
        self.current_context = {
            'dataset': None,
            'last_analysis': None,
            'active_columns': [],
            'user_preferences': {}
        }
    
    def _initialize_nlp_components(self):
        """Initialize NLP components for query processing"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers unavailable; skipping QA pipeline initialization")
                self.question_answerer = None
                return
            # Initialize question-answering pipeline
            self.question_answerer = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
            logger.info("NLP components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NLP components: {str(e)}")
            self.question_answerer = None
    
    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize common query patterns for intent recognition"""
        return {
            'summary': [
                r'(show|give|provide).*(summary|overview|description)',
                r'(what|how).*(data|dataset|information)',
                r'(describe|summarize|explain).*(data|dataset)',
                r'tell me about.*'
            ],
            'statistics': [
                r'(calculate|compute|find).*(mean|average|median|mode|std|statistics)',
                r'(what|how).*(average|mean|median|maximum|minimum)',
                r'(show|display).*(stats|statistics|numbers)',
                r'(get|find).*(distribution|spread)'
            ],
            'filtering': [
                r'(show|find|get|filter).*(where|with|having)',
                r'(records|rows|data).*(greater|less|equal|above|below)',
                r'(filter|select).*(by|on|where)',
                r'(find|show).*(all|records|data).*(that|which)'
            ],
            'comparison': [
                r'(compare|contrast|difference|versus|vs)',
                r'(which|what).*(higher|lower|better|worse|more|less)',
                r'(relationship|correlation|association).*(between|among)',
                r'(how).*(related|connected|associated)'
            ],
            'prediction': [
                r'(predict|forecast|estimate|project)',
                r'(what|how).*(will|would|might|could).*be',
                r'(future|next|upcoming|expected).*value',
                r'(model|algorithm).*(predict|forecast)'
            ],
            'visualization': [
                r'(plot|chart|graph|visualize|show)',
                r'(create|make|generate).*(plot|chart|graph)',
                r'(display|show).*(visualization|chart|graph)',
                r'(histogram|scatter|bar|line).*(plot|chart)'
            ],
            'aggregation': [
                r'(group|aggregate|sum|count|total)',
                r'(by|per|for each).*(category|group|type)',
                r'(count|number).*(of|per|by)',
                r'(total|sum|aggregate).*(by|per|for)'
            ]
        }
    
    def process_query(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a natural language query and return structured response
        
        Args:
            query: Natural language query from user
            data: DataFrame to query against
            
        Returns:
            Dictionary containing query results and metadata
        """
        try:
            # Update context
            self.current_context['dataset'] = data
            
            # Add to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'type': 'user_input'
            })
            
            # Parse and understand the query
            parsed_query = self._parse_query(query, data)
            
            # Execute the query
            result = self._execute_query(parsed_query, data)
            
            # Generate natural language response
            response = self._generate_response(result, query)
            
            # Add response to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'response': response,
                'result': result,
                'type': 'system_response'
            })
            
            return {
                'query': query,
                'parsed_query': parsed_query,
                'result': result,
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_response = f"I'm sorry, I encountered an error processing your query: {str(e)}"
            
            return {
                'query': query,
                'error': str(e),
                'response': error_response,
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
    
    def _parse_query(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Parse natural language query into structured format"""
        query_lower = query.lower()
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Extract entities (column names, values, operations)
        entities = self._extract_entities(query_lower, data)
        
        # Extract conditions and filters
        conditions = self._extract_conditions(query_lower, data)
        
        # Extract aggregation operations
        aggregations = self._extract_aggregations(query_lower)
        
        return {
            'intent': intent,
            'entities': entities,
            'conditions': conditions,
            'aggregations': aggregations,
            'original_query': query
        }
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query using pattern matching"""
        for intent, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        # Default intent if no pattern matches
        return 'general'
    
    def _extract_entities(self, query: str, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Extract entities like column names and values from query"""
        entities = {
            'columns': [],
            'values': [],
            'operations': []
        }
        
        # Extract column names
        for col in data.columns:
            if col.lower() in query or col.replace('_', ' ').lower() in query:
                entities['columns'].append(col)
        
        # Extract numeric values
        numeric_values = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        entities['values'].extend(numeric_values)
        
        # Extract operations
        operations = ['sum', 'count', 'average', 'mean', 'max', 'min', 'median', 'std']
        for op in operations:
            if op in query:
                entities['operations'].append(op)
        
        return entities
    
    def _extract_conditions(self, query: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract filtering conditions from query"""
        conditions = []
        
        # Pattern for conditions like "where column > value"
        condition_patterns = [
            r'(\w+)\s*(>|<|>=|<=|=|==|!=)\s*(\d+(?:\.\d+)?)',
            r'(\w+)\s*(greater|less|equal)\s*(?:than|to)?\s*(\d+(?:\.\d+)?)',
            r'(\w+)\s*(above|below|over|under)\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                column, operator, value = match
                
                # Map text operators to symbols
                operator_map = {
                    'greater': '>',
                    'less': '<',
                    'equal': '==',
                    'above': '>',
                    'below': '<',
                    'over': '>',
                    'under': '<'
                }
                
                if operator in operator_map:
                    operator = operator_map[operator]
                
                # Check if column exists in data
                matching_cols = [col for col in data.columns if col.lower() == column.lower()]
                if matching_cols:
                    conditions.append({
                        'column': matching_cols[0],
                        'operator': operator,
                        'value': float(value) if '.' in value else int(value)
                    })
        
        return conditions
    
    def _extract_aggregations(self, query: str) -> List[str]:
        """Extract aggregation operations from query"""
        aggregations = []
        
        agg_keywords = {
            'sum': ['sum', 'total', 'add'],
            'count': ['count', 'number', 'how many'],
            'mean': ['average', 'mean'],
            'median': ['median', 'middle'],
            'max': ['maximum', 'max', 'highest', 'largest'],
            'min': ['minimum', 'min', 'lowest', 'smallest'],
            'std': ['standard deviation', 'std', 'deviation']
        }
        
        for agg_func, keywords in agg_keywords.items():
            if any(keyword in query for keyword in keywords):
                aggregations.append(agg_func)
        
        return aggregations
    
    def _execute_query(self, parsed_query: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Execute the parsed query on the data"""
        intent = parsed_query['intent']
        entities = parsed_query['entities']
        conditions = parsed_query['conditions']
        aggregations = parsed_query['aggregations']
        
        result = {
            'intent': intent,
            'data': None,
            'summary': {},
            'visualizations': []
        }
        
        # Apply filters first
        filtered_data = data.copy()
        for condition in conditions:
            col = condition['column']
            op = condition['operator']
            val = condition['value']
            
            if op == '>':
                filtered_data = filtered_data[filtered_data[col] > val]
            elif op == '<':
                filtered_data = filtered_data[filtered_data[col] < val]
            elif op == '>=':
                filtered_data = filtered_data[filtered_data[col] >= val]
            elif op == '<=':
                filtered_data = filtered_data[filtered_data[col] <= val]
            elif op in ['=', '==']:
                filtered_data = filtered_data[filtered_data[col] == val]
            elif op == '!=':
                filtered_data = filtered_data[filtered_data[col] != val]
        
        # Execute based on intent
        if intent == 'summary':
            result = self._execute_summary_query(filtered_data, entities)
        elif intent == 'statistics':
            result = self._execute_statistics_query(filtered_data, entities, aggregations)
        elif intent == 'filtering':
            result = self._execute_filtering_query(filtered_data, entities)
        elif intent == 'comparison':
            result = self._execute_comparison_query(filtered_data, entities)
        elif intent == 'aggregation':
            result = self._execute_aggregation_query(filtered_data, entities, aggregations)
        elif intent == 'visualization':
            result = self._execute_visualization_query(filtered_data, entities)
        else:
            result = self._execute_general_query(filtered_data, entities)
        
        return result
    
    def _execute_summary_query(self, data: pd.DataFrame, entities: Dict) -> Dict[str, Any]:
        """Execute summary-type queries"""
        return {
            'intent': 'summary',
            'data': data.head(10).to_dict('records'),
            'summary': {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'columns': list(data.columns),
                'data_types': data.dtypes.astype(str).to_dict(),
                'missing_values': data.isnull().sum().to_dict()
            }
        }
    
    def _execute_statistics_query(self, data: pd.DataFrame, entities: Dict, aggregations: List[str]) -> Dict[str, Any]:
        """Execute statistics-type queries"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        target_cols = entities['columns'] if entities['columns'] else numeric_cols
        
        stats = {}
        for col in target_cols:
            if col in numeric_cols:
                col_stats = {}
                if not aggregations or 'mean' in aggregations:
                    col_stats['mean'] = data[col].mean()
                if not aggregations or 'median' in aggregations:
                    col_stats['median'] = data[col].median()
                if not aggregations or 'std' in aggregations:
                    col_stats['std'] = data[col].std()
                if not aggregations or 'min' in aggregations:
                    col_stats['min'] = data[col].min()
                if not aggregations or 'max' in aggregations:
                    col_stats['max'] = data[col].max()
                if not aggregations or 'count' in aggregations:
                    col_stats['count'] = data[col].count()
                
                stats[col] = col_stats
        
        return {
            'intent': 'statistics',
            'statistics': stats,
            'summary': f"Calculated statistics for {len(stats)} columns"
        }
    
    def _execute_filtering_query(self, data: pd.DataFrame, entities: Dict) -> Dict[str, Any]:
        """Execute filtering-type queries"""
        return {
            'intent': 'filtering',
            'data': data.head(20).to_dict('records'),
            'summary': {
                'filtered_rows': len(data),
                'original_rows': len(self.current_context['dataset']) if self.current_context['dataset'] is not None else len(data),
                'columns_shown': list(data.columns)
            }
        }
    
    def _execute_comparison_query(self, data: pd.DataFrame, entities: Dict) -> Dict[str, Any]:
        """Execute comparison-type queries"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        target_cols = entities['columns'] if entities['columns'] else numeric_cols[:2]
        
        comparisons = {}
        if len(target_cols) >= 2:
            col1, col2 = target_cols[0], target_cols[1]
            if col1 in numeric_cols and col2 in numeric_cols:
                correlation = data[col1].corr(data[col2])
                comparisons[f'{col1}_vs_{col2}'] = {
                    'correlation': correlation,
                    'col1_mean': data[col1].mean(),
                    'col2_mean': data[col2].mean(),
                    'relationship': 'positive' if correlation > 0.3 else 'negative' if correlation < -0.3 else 'weak'
                }
        
        return {
            'intent': 'comparison',
            'comparisons': comparisons,
            'summary': f"Compared {len(comparisons)} column pairs"
        }
    
    def _execute_aggregation_query(self, data: pd.DataFrame, entities: Dict, aggregations: List[str]) -> Dict[str, Any]:
        """Execute aggregation-type queries"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        results = {}
        
        # Group by categorical columns if available
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            group_col = categorical_cols[0]
            target_col = entities['columns'][0] if entities['columns'] and entities['columns'][0] in numeric_cols else numeric_cols[0]
            
            grouped = data.groupby(group_col)[target_col]
            
            for agg in aggregations or ['count', 'mean']:
                if hasattr(grouped, agg):
                    results[f'{agg}_by_{group_col}'] = getattr(grouped, agg)().to_dict()
        
        return {
            'intent': 'aggregation',
            'aggregations': results,
            'summary': f"Performed {len(results)} aggregation operations"
        }
    
    def _execute_visualization_query(self, data: pd.DataFrame, entities: Dict) -> Dict[str, Any]:
        """Execute visualization-type queries"""
        viz_suggestions = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) >= 2:
            viz_suggestions.append({
                'type': 'scatter',
                'x': numeric_cols[0],
                'y': numeric_cols[1],
                'description': f"Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}"
            })
        
        if len(categorical_cols) > 0:
            viz_suggestions.append({
                'type': 'bar',
                'x': categorical_cols[0],
                'description': f"Bar chart of {categorical_cols[0]} distribution"
            })
        
        if len(numeric_cols) > 0:
            viz_suggestions.append({
                'type': 'histogram',
                'x': numeric_cols[0],
                'description': f"Histogram of {numeric_cols[0]} distribution"
            })
        
        return {
            'intent': 'visualization',
            'visualizations': viz_suggestions,
            'summary': f"Suggested {len(viz_suggestions)} visualizations"
        }
    
    def _execute_general_query(self, data: pd.DataFrame, entities: Dict) -> Dict[str, Any]:
        """Execute general queries"""
        return {
            'intent': 'general',
            'data': data.head(5).to_dict('records'),
            'summary': {
                'message': "I've analyzed your data. Here's a sample of the results.",
                'rows_shown': min(5, len(data)),
                'total_rows': len(data)
            }
        }
    
    def _generate_response(self, result: Dict[str, Any], original_query: str) -> str:
        """Generate natural language response based on query results"""
        intent = result.get('intent', 'general')
        
        if intent == 'summary':
            return self._generate_summary_response(result)
        elif intent == 'statistics':
            return self._generate_statistics_response(result)
        elif intent == 'filtering':
            return self._generate_filtering_response(result)
        elif intent == 'comparison':
            return self._generate_comparison_response(result)
        elif intent == 'aggregation':
            return self._generate_aggregation_response(result)
        elif intent == 'visualization':
            return self._generate_visualization_response(result)
        else:
            return self._generate_general_response(result)
    
    def _generate_summary_response(self, result: Dict[str, Any]) -> str:
        """Generate response for summary queries"""
        summary = result.get('summary', {})
        total_rows = summary.get('total_rows', 0)
        total_cols = summary.get('total_columns', 0)
        
        response = f"Your dataset contains {total_rows} rows and {total_cols} columns. "
        
        missing_values = summary.get('missing_values', {})
        missing_count = sum(v for v in missing_values.values() if v > 0)
        if missing_count > 0:
            response += f"There are {missing_count} missing values across the dataset. "
        
        response += "I've shown you a sample of the data above."
        return response
    
    def _generate_statistics_response(self, result: Dict[str, Any]) -> str:
        """Generate response for statistics queries"""
        stats = result.get('statistics', {})
        
        if not stats:
            return "I couldn't calculate statistics for the requested columns."
        
        response = f"I've calculated statistics for {len(stats)} columns. "
        
        # Highlight interesting findings
        for col, col_stats in stats.items():
            if 'mean' in col_stats:
                response += f"The average {col} is {col_stats['mean']:.2f}. "
                break  # Just show one example to keep response concise
        
        return response
    
    def _generate_filtering_response(self, result: Dict[str, Any]) -> str:
        """Generate response for filtering queries"""
        summary = result.get('summary', {})
        filtered_rows = summary.get('filtered_rows', 0)
        original_rows = summary.get('original_rows', 0)
        
        if filtered_rows == original_rows:
            return f"Your query returned all {filtered_rows} rows from the dataset."
        else:
            return f"Your filter returned {filtered_rows} rows out of {original_rows} total rows."
    
    def _generate_comparison_response(self, result: Dict[str, Any]) -> str:
        """Generate response for comparison queries"""
        comparisons = result.get('comparisons', {})
        
        if not comparisons:
            return "I couldn't find meaningful comparisons in your data."
        
        response = "Here's what I found when comparing your data: "
        
        for comparison_name, comparison_data in comparisons.items():
            correlation = comparison_data.get('correlation', 0)
            relationship = comparison_data.get('relationship', 'unknown')
            
            response += f"The correlation shows a {relationship} relationship (r={correlation:.3f}). "
            break  # Keep response concise
        
        return response
    
    def _generate_aggregation_response(self, result: Dict[str, Any]) -> str:
        """Generate response for aggregation queries"""
        aggregations = result.get('aggregations', {})
        
        if not aggregations:
            return "I couldn't perform the requested aggregation on your data."
        
        response = f"I've performed {len(aggregations)} aggregation operations on your data. "
        response += "The results show the grouped statistics you requested."
        
        return response
    
    def _generate_visualization_response(self, result: Dict[str, Any]) -> str:
        """Generate response for visualization queries"""
        visualizations = result.get('visualizations', [])
        
        if not visualizations:
            return "I couldn't suggest appropriate visualizations for your data."
        
        response = f"I've suggested {len(visualizations)} visualizations for your data: "
        
        for viz in visualizations[:2]:  # Show first 2 suggestions
            response += f"{viz['description']}. "
        
        return response
    
    def _generate_general_response(self, result: Dict[str, Any]) -> str:
        """Generate response for general queries"""
        summary = result.get('summary', {})
        message = summary.get('message', "I've processed your query and analyzed the data.")
        
        return message
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    def get_query_suggestions(self, data: pd.DataFrame) -> List[str]:
        """Generate query suggestions based on the current dataset"""
        suggestions = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Basic exploration suggestions
        suggestions.append("Show me a summary of this data")
        suggestions.append("What are the main statistics?")
        
        # Column-specific suggestions
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            suggestions.append(f"What is the average {col}?")
            suggestions.append(f"Show me the distribution of {col}")
        
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            suggestions.append(f"How many different {col} categories are there?")
        
        # Relationship suggestions
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            suggestions.append(f"What is the relationship between {col1} and {col2}?")
        
        # Filtering suggestions
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            median_val = data[col].median()
            suggestions.append(f"Show me records where {col} is greater than {median_val}")
        
        return suggestions[:6]  # Return top 6 suggestions
    
    def process_follow_up_query(self, query: str) -> Dict[str, Any]:
        """Process follow-up queries that reference previous results"""
        # This is a simplified implementation
        # In a full system, this would maintain more sophisticated context
        
        if not self.conversation_history:
            return {
                'error': 'No previous conversation to reference',
                'suggestion': 'Please start with a new query about your data'
            }
        
        last_response = self.conversation_history[-1]
        
        # Simple context-aware processing
        if 'more' in query.lower() or 'additional' in query.lower():
            return {
                'message': 'I can provide more details about the previous analysis',
                'context': last_response.get('result', {}),
                'suggestion': 'Please specify what additional information you need'
            }
        
        return {
            'message': 'I understand you\'re asking a follow-up question',
            'suggestion': 'Please rephrase your question with more context'
        }