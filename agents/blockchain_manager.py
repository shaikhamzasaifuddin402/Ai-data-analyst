"""
Blockchain Manager for Data Integrity and Audit Trails
Advanced blockchain integration for trusted record keeping and smart contracts
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class Block:
    """Individual block in the blockchain"""
    
    def __init__(self, index: int, data: Dict[str, Any], previous_hash: str):
        self.index = index
        self.timestamp = time.time()
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate the hash of the block"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4):
        """Mine the block with proof of work"""
        target = "0" * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        
        return self.hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            'data': self.data,
            'previous_hash': self.previous_hash,
            'hash': self.hash,
            'nonce': self.nonce
        }

class SmartContract:
    """Smart contract for automated data validation and processing"""
    
    def __init__(self, contract_id: str, rules: Dict[str, Any]):
        self.contract_id = contract_id
        self.rules = rules
        self.execution_history = []
        self.created_at = time.time()
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against contract rules"""
        validation_result = {
            'valid': True,
            'violations': [],
            'warnings': [],
            'metadata': {}
        }
        
        # Check required fields
        if 'required_fields' in self.rules:
            for field in self.rules['required_fields']:
                if field not in data:
                    validation_result['valid'] = False
                    validation_result['violations'].append(f"Missing required field: {field}")
        
        # Check data types
        if 'data_types' in self.rules:
            for field, expected_type in self.rules['data_types'].items():
                if field in data:
                    if expected_type == 'numeric' and not isinstance(data[field], (int, float)):
                        validation_result['valid'] = False
                        validation_result['violations'].append(f"Field {field} must be numeric")
                    elif expected_type == 'string' and not isinstance(data[field], str):
                        validation_result['valid'] = False
                        validation_result['violations'].append(f"Field {field} must be string")
        
        # Check value ranges
        if 'value_ranges' in self.rules:
            for field, range_rule in self.rules['value_ranges'].items():
                if field in data and isinstance(data[field], (int, float)):
                    if 'min' in range_rule and data[field] < range_rule['min']:
                        validation_result['valid'] = False
                        validation_result['violations'].append(f"Field {field} below minimum: {range_rule['min']}")
                    if 'max' in range_rule and data[field] > range_rule['max']:
                        validation_result['valid'] = False
                        validation_result['violations'].append(f"Field {field} above maximum: {range_rule['max']}")
        
        # Check custom rules
        if 'custom_rules' in self.rules:
            for rule_name, rule_func in self.rules['custom_rules'].items():
                try:
                    if not rule_func(data):
                        validation_result['valid'] = False
                        validation_result['violations'].append(f"Custom rule violation: {rule_name}")
                except Exception as e:
                    validation_result['warnings'].append(f"Error in custom rule {rule_name}: {str(e)}")
        
        # Record execution
        execution_record = {
            'timestamp': time.time(),
            'data_hash': hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
            'result': validation_result,
            'contract_id': self.contract_id
        }
        self.execution_history.append(execution_record)
        
        return validation_result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary"""
        return {
            'contract_id': self.contract_id,
            'rules': self.rules,
            'created_at': self.created_at,
            'execution_count': len(self.execution_history)
        }

class DataBlockchain:
    """Blockchain for data integrity and audit trails"""
    
    def __init__(self):
        self.chain = []
        self.difficulty = 4
        self.smart_contracts = {}
        self.pending_transactions = []
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_data = {
            'type': 'genesis',
            'message': 'Genesis block for Data Analytics Platform',
            'timestamp': time.time()
        }
        genesis_block = Block(0, genesis_data, "0")
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the latest block in the chain"""
        return self.chain[-1]
    
    def add_data_record(self, data: Dict[str, Any], record_type: str = 'data_entry') -> str:
        """Add a data record to the blockchain"""
        # Create data hash for integrity
        data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
        block_data = {
            'type': record_type,
            'data': data,
            'data_hash': data_hash,
            'timestamp': time.time(),
            'user': 'system'  # In production, this would be the authenticated user
        }
        
        # Validate with smart contracts if applicable
        validation_results = {}
        for contract_id, contract in self.smart_contracts.items():
            validation_results[contract_id] = contract.validate_data(data)
        
        if validation_results:
            block_data['validations'] = validation_results
        
        # Create and mine new block
        new_block = Block(
            len(self.chain),
            block_data,
            self.get_latest_block().hash
        )
        new_block.mine_block(self.difficulty)
        
        # Validate chain integrity before adding
        if self.is_chain_valid():
            self.chain.append(new_block)
            return new_block.hash
        else:
            raise Exception("Blockchain integrity compromised - cannot add new block")
    
    def add_smart_contract(self, contract: SmartContract):
        """Add a smart contract to the blockchain"""
        contract_data = {
            'type': 'smart_contract',
            'contract': contract.to_dict(),
            'action': 'deploy'
        }
        
        self.smart_contracts[contract.contract_id] = contract
        return self.add_data_record(contract_data, 'smart_contract_deployment')
    
    def is_chain_valid(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block's hash is valid
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Check if current block points to previous block
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_audit_trail(self, data_hash: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit trail for specific data or entire chain"""
        audit_trail = []
        
        for block in self.chain:
            block_dict = block.to_dict()
            
            # Filter by data hash if specified
            if data_hash is None or block_dict['data'].get('data_hash') == data_hash:
                audit_trail.append(block_dict)
        
        return audit_trail
    
    def get_chain_statistics(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        total_blocks = len(self.chain)
        total_data_records = sum(1 for block in self.chain if block.data.get('type') == 'data_entry')
        total_contracts = len(self.smart_contracts)
        
        # Calculate chain integrity score
        integrity_score = 100.0 if self.is_chain_valid() else 0.0
        
        # Calculate average block time
        if total_blocks > 1:
            time_diff = self.chain[-1].timestamp - self.chain[0].timestamp
            avg_block_time = time_diff / (total_blocks - 1)
        else:
            avg_block_time = 0
        
        return {
            'total_blocks': total_blocks,
            'total_data_records': total_data_records,
            'total_smart_contracts': total_contracts,
            'integrity_score': integrity_score,
            'average_block_time': avg_block_time,
            'chain_size_kb': len(json.dumps([block.to_dict() for block in self.chain])) / 1024,
            'latest_block_hash': self.get_latest_block().hash
        }

class BlockchainManager:
    """Main blockchain manager for the analytics platform"""
    
    def __init__(self):
        self.blockchain = DataBlockchain()
        self.data_registry = {}  # Maps data IDs to blockchain hashes
        self.access_logs = []
    
    def register_dataset(self, dataset: pd.DataFrame, dataset_id: str, metadata: Dict[str, Any] = None) -> str:
        """Register a dataset on the blockchain"""
        # Create dataset fingerprint
        dataset_hash = hashlib.sha256(str(dataset.values.tobytes()).encode()).hexdigest()
        
        dataset_record = {
            'dataset_id': dataset_id,
            'dataset_hash': dataset_hash,
            'shape': dataset.shape,
            'columns': list(dataset.columns),
            'dtypes': {col: str(dtype) for col, dtype in dataset.dtypes.items()},
            'metadata': metadata or {},
            'registered_by': 'system'
        }
        
        # Add to blockchain
        block_hash = self.blockchain.add_data_record(dataset_record, 'dataset_registration')
        
        # Update registry
        self.data_registry[dataset_id] = {
            'block_hash': block_hash,
            'dataset_hash': dataset_hash,
            'registered_at': time.time()
        }
        
        return block_hash
    
    def verify_dataset_integrity(self, dataset: pd.DataFrame, dataset_id: str) -> Dict[str, Any]:
        """Verify dataset integrity against blockchain record"""
        if dataset_id not in self.data_registry:
            return {'verified': False, 'error': 'Dataset not found in registry'}
        
        # Calculate current dataset hash
        current_hash = hashlib.sha256(str(dataset.values.tobytes()).encode()).hexdigest()
        registered_hash = self.data_registry[dataset_id]['dataset_hash']
        
        verification_result = {
            'verified': current_hash == registered_hash,
            'current_hash': current_hash,
            'registered_hash': registered_hash,
            'dataset_id': dataset_id,
            'verification_time': time.time()
        }
        
        # Log access
        self.access_logs.append({
            'action': 'verify_integrity',
            'dataset_id': dataset_id,
            'result': verification_result['verified'],
            'timestamp': time.time()
        })
        
        return verification_result
    
    def create_data_contract(self, contract_id: str, rules: Dict[str, Any]) -> str:
        """Create a smart contract for data validation"""
        contract = SmartContract(contract_id, rules)
        return self.blockchain.add_smart_contract(contract)
    
    def get_comprehensive_audit_trail(self) -> pd.DataFrame:
        """Get comprehensive audit trail as DataFrame"""
        audit_data = []
        
        for block in self.blockchain.chain:
            block_dict = block.to_dict()
            
            audit_record = {
                'block_index': block_dict['index'],
                'timestamp': block_dict['timestamp'],
                'datetime': block_dict['datetime'],
                'block_hash': block_dict['hash'],
                'previous_hash': block_dict['previous_hash'],
                'data_type': block_dict['data'].get('type', 'unknown'),
                'nonce': block_dict['nonce'],
                'dataset_id': None,  # Initialize with None for all records
                'dataset_shape': None,
                'columns_count': None
            }
            
            # Add specific data based on type
            if block_dict['data'].get('type') == 'dataset_registration':
                audit_record.update({
                    'dataset_id': block_dict['data']['data'].get('dataset_id'),
                    'dataset_shape': str(block_dict['data']['data'].get('shape')),
                    'columns_count': len(block_dict['data']['data'].get('columns', []))
                })
            
            audit_data.append(audit_record)
        
        return pd.DataFrame(audit_data)

def render_blockchain_manager():
    """Render the Blockchain Manager interface"""
    st.header("🔗 Blockchain Data Integrity Manager")
    st.markdown("**Secure Data Registry & Audit Trails with Smart Contracts**")
    
    # Initialize blockchain manager
    if 'blockchain_manager' not in st.session_state:
        st.session_state.blockchain_manager = BlockchainManager()
    
    manager = st.session_state.blockchain_manager
    blockchain = manager.blockchain
    
    # Blockchain status dashboard
    st.subheader("📊 Blockchain Status Dashboard")
    
    stats = blockchain.get_chain_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Blocks", stats['total_blocks'])
    
    with col2:
        st.metric("Data Records", stats['total_data_records'])
    
    with col3:
        st.metric("Smart Contracts", stats['total_smart_contracts'])
    
    with col4:
        integrity_color = "normal" if stats['integrity_score'] == 100 else "inverse"
        st.metric("Chain Integrity", f"{stats['integrity_score']:.1f}%", delta_color=integrity_color)
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Chain Size", f"{stats['chain_size_kb']:.1f} KB")
    
    with col2:
        st.metric("Avg Block Time", f"{stats['average_block_time']:.2f}s")
    
    with col3:
        st.metric("Latest Block", stats['latest_block_hash'][:16] + "...")
    
    # Main functionality tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Dataset Registration",
        "🔍 Integrity Verification", 
        "📋 Smart Contracts",
        "📊 Audit Trail",
        "⚙️ Blockchain Explorer"
    ])
    
    with tab1:
        render_dataset_registration(manager)
    
    with tab2:
        render_integrity_verification(manager)
    
    with tab3:
        render_smart_contracts(manager)
    
    with tab4:
        render_audit_trail(manager)
    
    with tab5:
        render_blockchain_explorer(manager)

def render_dataset_registration(manager):
    """Render dataset registration interface"""
    st.subheader("📝 Dataset Registration")
    st.markdown("Register datasets on the blockchain for integrity tracking")
    
    # File upload
    uploaded_file = st.file_uploader("Upload dataset to register", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Load dataset
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Dataset preview
            with st.expander("📋 Dataset Preview"):
                st.dataframe(df.head())
            
            # Registration form
            st.markdown("**Registration Details:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                dataset_id = st.text_input("Dataset ID:", value=uploaded_file.name.split('.')[0])
            
            with col2:
                dataset_version = st.text_input("Version:", value="1.0")
            
            # Metadata
            st.markdown("**Metadata (Optional):**")
            description = st.text_area("Description:")
            source = st.text_input("Data Source:")
            owner = st.text_input("Data Owner:")
            
            # Register dataset
            if st.button("🔗 Register on Blockchain"):
                metadata = {
                    'version': dataset_version,
                    'description': description,
                    'source': source,
                    'owner': owner,
                    'file_name': uploaded_file.name,
                    'registration_timestamp': time.time()
                }
                
                with st.spinner("Registering dataset on blockchain..."):
                    try:
                        block_hash = manager.register_dataset(df, dataset_id, metadata)
                        st.success(f"✅ Dataset registered successfully!")
                        st.info(f"**Block Hash:** `{block_hash}`")
                        st.info(f"**Dataset ID:** `{dataset_id}`")
                        
                        # Show registration details
                        st.markdown("**Registration Summary:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"• **Rows:** {df.shape[0]:,}")
                            st.write(f"• **Columns:** {df.shape[1]}")
                            st.write(f"• **Size:** {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                        
                        with col2:
                            st.write(f"• **Version:** {dataset_version}")
                            st.write(f"• **Owner:** {owner or 'Not specified'}")
                            st.write(f"• **Source:** {source or 'Not specified'}")
                        
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    # Show registered datasets
    if manager.data_registry:
        st.markdown("---")
        st.subheader("📚 Registered Datasets")
        
        registry_data = []
        for dataset_id, info in manager.data_registry.items():
            registry_data.append({
                'Dataset ID': dataset_id,
                'Block Hash': info['block_hash'][:16] + "...",
                'Dataset Hash': info['dataset_hash'][:16] + "...",
                'Registered At': datetime.fromtimestamp(info['registered_at']).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        registry_df = pd.DataFrame(registry_data)
        st.dataframe(registry_df, use_container_width=True)

def render_integrity_verification(manager):
    """Render integrity verification interface"""
    st.subheader("🔍 Dataset Integrity Verification")
    st.markdown("Verify dataset integrity against blockchain records")
    
    if not manager.data_registry:
        st.info("No datasets registered yet. Please register a dataset first.")
        return
    
    # Dataset selection
    dataset_ids = list(manager.data_registry.keys())
    selected_dataset = st.selectbox("Select dataset to verify:", dataset_ids)
    
    # File upload for verification
    uploaded_file = st.file_uploader("Upload dataset file for verification", type=['csv', 'xlsx'])
    
    if uploaded_file is not None and selected_dataset:
        try:
            # Load dataset
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.info(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Verify integrity
            if st.button("🔍 Verify Integrity"):
                with st.spinner("Verifying dataset integrity..."):
                    verification_result = manager.verify_dataset_integrity(df, selected_dataset)
                    
                    if verification_result['verified']:
                        st.success("✅ **Dataset Integrity Verified!**")
                        st.success("The dataset matches the blockchain record exactly.")
                    else:
                        st.error("❌ **Integrity Verification Failed!**")
                        st.error("The dataset has been modified since registration.")
                    
                    # Show verification details
                    st.markdown("**Verification Details:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Current Hash:**")
                        st.code(verification_result['current_hash'])
                    
                    with col2:
                        st.write("**Registered Hash:**")
                        st.code(verification_result['registered_hash'])
                    
                    # Show verification timestamp
                    verification_time = datetime.fromtimestamp(verification_result['verification_time'])
                    st.info(f"Verification performed at: {verification_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        except Exception as e:
            st.error(f"Error verifying dataset: {str(e)}")
    
    # Show verification history
    if manager.access_logs:
        st.markdown("---")
        st.subheader("📋 Verification History")
        
        verification_logs = [log for log in manager.access_logs if log['action'] == 'verify_integrity']
        
        if verification_logs:
            log_data = []
            for log in verification_logs[-10:]:  # Show last 10 verifications
                log_data.append({
                    'Dataset ID': log['dataset_id'],
                    'Result': '✅ Verified' if log['result'] else '❌ Failed',
                    'Timestamp': datetime.fromtimestamp(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            log_df = pd.DataFrame(log_data)
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("No verification history available.")

def render_smart_contracts(manager):
    """Render smart contracts interface"""
    st.subheader("📋 Smart Contracts")
    st.markdown("Create and manage smart contracts for automated data validation")
    
    # Contract creation
    st.markdown("**Create New Smart Contract:**")
    
    contract_id = st.text_input("Contract ID:", placeholder="e.g., data_quality_validator")
    
    if contract_id:
        st.markdown("**Contract Rules:**")
        
        # Required fields
        required_fields = st.text_input("Required Fields (comma-separated):", 
                                      placeholder="e.g., id, name, value")
        
        # Data type rules
        st.markdown("**Data Type Rules:**")
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_fields = st.text_input("Numeric Fields:", placeholder="e.g., age, salary, score")
        
        with col2:
            string_fields = st.text_input("String Fields:", placeholder="e.g., name, category, status")
        
        # Value range rules
        st.markdown("**Value Range Rules:**")
        range_field = st.text_input("Field Name:", placeholder="e.g., age")
        
        if range_field:
            col1, col2 = st.columns(2)
            with col1:
                min_value = st.number_input("Minimum Value:", value=0.0)
            with col2:
                max_value = st.number_input("Maximum Value:", value=100.0)
        
        # Create contract
        if st.button("📋 Deploy Smart Contract"):
            try:
                # Build rules dictionary
                rules = {}
                
                if required_fields:
                    rules['required_fields'] = [field.strip() for field in required_fields.split(',')]
                
                data_types = {}
                if numeric_fields:
                    for field in numeric_fields.split(','):
                        data_types[field.strip()] = 'numeric'
                if string_fields:
                    for field in string_fields.split(','):
                        data_types[field.strip()] = 'string'
                
                if data_types:
                    rules['data_types'] = data_types
                
                if range_field and min_value is not None and max_value is not None:
                    rules['value_ranges'] = {
                        range_field: {'min': min_value, 'max': max_value}
                    }
                
                # Deploy contract
                block_hash = manager.create_data_contract(contract_id, rules)
                st.success(f"✅ Smart contract deployed successfully!")
                st.info(f"**Block Hash:** `{block_hash}`")
                st.info(f"**Contract ID:** `{contract_id}`")
                
            except Exception as e:
                st.error(f"Contract deployment failed: {str(e)}")
    
    # Show existing contracts
    if manager.blockchain.smart_contracts:
        st.markdown("---")
        st.subheader("📚 Deployed Smart Contracts")
        
        for contract_id, contract in manager.blockchain.smart_contracts.items():
            with st.expander(f"📋 Contract: {contract_id}"):
                contract_dict = contract.to_dict()
                
                st.write(f"**Created:** {datetime.fromtimestamp(contract_dict['created_at']).strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Executions:** {contract_dict['execution_count']}")
                
                st.markdown("**Rules:**")
                st.json(contract_dict['rules'])
                
                # Test contract
                st.markdown("**Test Contract:**")
                test_data = st.text_area(f"Test data (JSON format):", 
                                       placeholder='{"field1": "value1", "field2": 123}',
                                       key=f"test_{contract_id}")
                
                if st.button(f"🧪 Test Contract", key=f"test_btn_{contract_id}"):
                    try:
                        test_dict = json.loads(test_data)
                        result = contract.validate_data(test_dict)
                        
                        if result['valid']:
                            st.success("✅ Validation passed!")
                        else:
                            st.error("❌ Validation failed!")
                            for violation in result['violations']:
                                st.error(f"• {violation}")
                        
                        if result['warnings']:
                            for warning in result['warnings']:
                                st.warning(f"⚠️ {warning}")
                    
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format")
                    except Exception as e:
                        st.error(f"Test failed: {str(e)}")

def render_audit_trail(manager):
    """Render audit trail interface"""
    st.subheader("📊 Comprehensive Audit Trail")
    st.markdown("Complete blockchain audit trail and analytics")
    
    # Get audit trail
    audit_df = manager.get_comprehensive_audit_trail()
    
    if len(audit_df) > 0:
        # Audit trail summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", len(audit_df))
        
        with col2:
            data_records = len(audit_df[audit_df['data_type'] == 'dataset_registration'])
            st.metric("Dataset Registrations", data_records)
        
        with col3:
            contract_records = len(audit_df[audit_df['data_type'] == 'smart_contract'])
            st.metric("Contract Deployments", contract_records)
        
        # Timeline visualization
        st.subheader("📈 Transaction Timeline")
        
        # Convert timestamp to datetime for plotting
        audit_df['datetime_parsed'] = pd.to_datetime(audit_df['timestamp'], unit='s')
        
        # Create timeline chart
        # Only include hover_data columns that exist in the dataframe
        available_hover_data = []
        if 'block_hash' in audit_df.columns:
            available_hover_data.append('block_hash')
        if 'dataset_id' in audit_df.columns:
            available_hover_data.append('dataset_id')
        
        fig = px.scatter(
            audit_df, 
            x='datetime_parsed', 
            y='block_index',
            color='data_type',
            hover_data=available_hover_data,
            title="Blockchain Transaction Timeline"
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Block Index")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed audit trail
        st.subheader("📋 Detailed Audit Trail")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            data_type_filter = st.selectbox("Filter by Type:", 
                                          ['All'] + list(audit_df['data_type'].unique()))
        
        with col2:
            show_last_n = st.slider("Show last N records:", 5, 50, 20)
        
        # Apply filters
        filtered_df = audit_df.copy()
        
        if data_type_filter != 'All':
            filtered_df = filtered_df[filtered_df['data_type'] == data_type_filter]
        
        # Show filtered results
        display_df = filtered_df.tail(show_last_n).sort_values('block_index', ascending=False)
        st.dataframe(display_df, use_container_width=True)
        
        # Export audit trail
        if st.button("📥 Export Audit Trail"):
            csv = audit_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"blockchain_audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("No transactions recorded yet. Register a dataset or deploy a smart contract to see audit trail.")

def render_blockchain_explorer(manager):
    """Render blockchain explorer interface"""
    st.subheader("⚙️ Blockchain Explorer")
    st.markdown("Explore individual blocks and chain structure")
    
    blockchain = manager.blockchain
    
    # Chain overview
    st.markdown("**Chain Overview:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Chain Length", len(blockchain.chain))
    
    with col2:
        is_valid = blockchain.is_chain_valid()
        st.metric("Chain Valid", "✅ Yes" if is_valid else "❌ No")
    
    with col3:
        st.metric("Mining Difficulty", blockchain.difficulty)
    
    # Block explorer
    st.markdown("---")
    st.subheader("🔍 Block Explorer")
    
    if len(blockchain.chain) > 0:
        # Block selection
        block_indices = list(range(len(blockchain.chain)))
        selected_block_idx = st.selectbox("Select Block:", block_indices, 
                                        format_func=lambda x: f"Block #{x}")
        
        if selected_block_idx is not None:
            block = blockchain.chain[selected_block_idx]
            block_dict = block.to_dict()
            
            # Block details
            st.markdown(f"**Block #{block_dict['index']} Details:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Hash:** `{block_dict['hash']}`")
                st.write(f"**Previous Hash:** `{block_dict['previous_hash']}`")
                st.write(f"**Timestamp:** {block_dict['datetime']}")
            
            with col2:
                st.write(f"**Nonce:** {block_dict['nonce']}")
                st.write(f"**Data Type:** {block_dict['data'].get('type', 'Unknown')}")
                
                # Verify block
                is_block_valid = block.hash == block.calculate_hash()
                st.write(f"**Valid:** {'✅ Yes' if is_block_valid else '❌ No'}")
            
            # Block data
            st.markdown("**Block Data:**")
            st.json(block_dict['data'])
            
            # Block connections visualization
            if len(blockchain.chain) > 1:
                st.markdown("---")
                st.subheader("🔗 Chain Visualization")
                
                # Create chain visualization
                chain_data = []
                for i, blk in enumerate(blockchain.chain):
                    chain_data.append({
                        'block': i,
                        'hash': blk.hash[:8] + "...",
                        'prev_hash': blk.previous_hash[:8] + "..." if blk.previous_hash != "0" else "Genesis",
                        'type': blk.data.get('type', 'unknown')
                    })
                
                chain_df = pd.DataFrame(chain_data)
                st.dataframe(chain_df, use_container_width=True)
    
    else:
        st.info("No blocks in the chain yet.")
    
    # Chain validation
    st.markdown("---")
    st.subheader("🔐 Chain Validation")
    
    if st.button("🔍 Validate Entire Chain"):
        with st.spinner("Validating blockchain..."):
            is_valid = blockchain.is_chain_valid()
            
            if is_valid:
                st.success("✅ Blockchain is valid and secure!")
            else:
                st.error("❌ Blockchain validation failed - integrity compromised!")
            
            # Detailed validation
            st.markdown("**Detailed Validation Results:**")
            
            for i in range(len(blockchain.chain)):
                block = blockchain.chain[i]
                
                # Check hash validity
                hash_valid = block.hash == block.calculate_hash()
                
                # Check chain linkage (except genesis block)
                if i > 0:
                    prev_block = blockchain.chain[i - 1]
                    link_valid = block.previous_hash == prev_block.hash
                else:
                    link_valid = True  # Genesis block
                
                status = "✅" if (hash_valid and link_valid) else "❌"
                st.write(f"{status} Block #{i}: Hash Valid: {hash_valid}, Link Valid: {link_valid}")