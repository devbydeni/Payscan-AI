"""
PayScan AI Agent - Bedrock Agent Processing Function
Handles final analysis and curation using Amazon Bedrock AgentCore
"""

import json
import boto3
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
bedrock_agent = boto3.client('bedrock-agent')
bedrock_runtime = boto3.client('bedrock-agent-runtime')
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

class PayScanBedrockProcessor:
    def __init__(self):
        self.agent_id = os.environ.get('BEDROCK_AGENT_ID')
        self.agent_alias_id = os.environ.get('BEDROCK_AGENT_ALIAS_ID', 'TSTALIASID')
        self.memory_table_name = os.environ.get('MEMORY_TABLE_NAME', 'payscan-agent-memory')
        self.session_ttl_hours = 24
        
        # Initialize DynamoDB table for memory persistence
        try:
            self.memory_table = dynamodb.Table(self.memory_table_name)
        except Exception as e:
            logger.warning(f"Could not initialize memory table: {str(e)}")
            self.memory_table = None
    
    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Main Lambda handler for Bedrock Agent processing"""
        try:
            logger.info(f"Starting Bedrock Agent processing: {json.dumps(event, default=str)}")
            
            invoice_data = event['invoice_data']
            textract_data = event['textract_data']
            job_id = event['job_id']
            processing_bucket = event['processing_bucket']
            
            # Validate input data
            if not invoice_data.get('success'):
                raise Exception("SageMaker processing was not successful")
            
            # Create session for this invoice processing
            session_id = f"payscan-{job_id}"
            
            # Process with Bedrock Agent in multiple stages
            processing_results = {}
            
            # Stage 1: Curate and validate the invoice data
            logger.info("Stage 1: Curating invoice data with Bedrock Agent")
            curation_result = self.curate_invoice_data(
                session_id, 
                invoice_data['structured_invoice_data'],
                textract_data['structured_data']
            )
            processing_results['curation'] = curation_result
            
            # Stage 2: Generate insights and alerts
            logger.info("Stage 2: Generating insights and alerts")
            insights_result = self.generate_insights_and_alerts(
                session_id,
                curation_result.get('curated_data', {})
            )
            processing_results['insights'] = insights_result
            
            # Stage 3: Prepare final recommendations
            logger.info("Stage 3: Preparing recommendations")
            recommendations_result = self.generate_recommendations(
                session_id,
                curation_result.get('curated_data', {}),
                insights_result.get('alerts', [])
            )
            processing_results['recommendations'] = recommendations_result
            
            # Combine all results
            final_result = self.combine_bedrock_results(
                invoice_data['structured_invoice_data'],
                processing_results,
                job_id
            )
            
            # Store results
            storage_key = f"bedrock/{job_id}/final_analysis.json"
            self.store_bedrock_results(processing_bucket, storage_key, final_result)
            
            # Store in agent memory for future reference
            self.store_in_agent_memory(session_id, final_result)
            
            return {
                'success': True,
                'job_id': job_id,
                'session_id': session_id,
                'agent_id': self.agent_id,
                'final_analysis_s3': f"s3://{processing_bucket}/{storage_key}",
                'curated_invoice_data': final_result,
                'processing_stats': {
                    'bedrock_stages_completed': len([r for r in processing_results.values() if r.get('success')]),
                    'total_alerts_generated': len(final_result.get('alerts', [])),
                    'risk_level': final_result.get('risk_assessment', {}).get('level', 'unknown')
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Bedrock Agent processing: {str(e)}")
            return {
                'success': False,
                'job_id': event.get('job_id'),
                'error': str(e),
                'error_type': 'bedrock_processing_error'
            }
    
    def curate_invoice_data(self, session_id: str, invoice_data: Dict[str, Any], textract_data: Dict[str, Any]) -> Dict[str, Any]:
        """Curate invoice data using Bedrock Agent"""
        try:
            # Prepare the prompt for curation
            curation_prompt = self.build_curation_prompt(invoice_data, textract_data)
            
            # Invoke Bedrock Agent
            response = self.invoke_bedrock_agent(session_id, curation_prompt)
            
            if response.get('success'):
                # Parse the agent's response
                curated_data = self.parse_curation_response(response['response_text'])
                
                return {
                    'success': True,
                    'curated_data': curated_data,
                    'agent_response': response['response_text'],
                    'session_id': session_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': response.get('error'),
                    'session_id': session_id
                }
                
        except Exception as e:
            logger.error(f"Error in curation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    def generate_insights_and_alerts(self, session_id: str, curated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights and alerts using Bedrock Agent"""
        try:
            # Build insights prompt
            insights_prompt = self.build_insights_prompt(curated_data)
            
            # Invoke Bedrock Agent
            response = self.invoke_bedrock_agent(session_id, insights_prompt)
            
            if response.get('success'):
                # Parse insights and alerts
                insights_data = self.parse_insights_response(response['response_text'])
                
                return {
                    'success': True,
                    'alerts': insights_data.get('alerts', []),
                    'insights': insights_data.get('insights', []),
                    'risk_factors': insights_data.get('risk_factors', []),
                    'agent_response': response['response_text'],
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': response.get('error')
                }
                
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_recommendations(self, session_id: str, curated_data: Dict[str, Any], alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate recommendations using Bedrock Agent"""
        try:
            # Build recommendations prompt
            recommendations_prompt = self.build_recommendations_prompt(curated_data, alerts)
            
            # Invoke Bedrock Agent
            response = self.invoke_bedrock_agent(session_id, recommendations_prompt)
            
            if response.get('success'):
                # Parse recommendations
                recommendations_data = self.parse_recommendations_response(response['response_text'])
                
                return {
                    'success': True,
                    'recommendations': recommendations_data.get('recommendations', []),
                    'next_actions': recommendations_data.get('next_actions', []),
                    'automation_suggestions': recommendations_data.get('automation_suggestions', []),
                    'agent_response': response['response_text'],
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': response.get('error')
                }
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def invoke_bedrock_agent(self, session_id: str, prompt: str) -> Dict[str, Any]:
        """Invoke the Bedrock Agent with the given prompt"""
        try:
            if not self.agent_id:
                raise Exception("Bedrock Agent ID not configured")
            
            logger.info(f"Invoking Bedrock Agent {self.agent_id} with session {session_id}")
            
            # Invoke the agent
            response = bedrock_runtime.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText=prompt
            )
            
            # Process the response stream
            response_text = ""
            event_stream = response.get('completion', {})
            
            for event in event_stream:
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        response_text += chunk['bytes'].decode('utf-8')
            
            logger.info(f"Received response from Bedrock Agent: {len(response_text)} characters")
            
            return {
                'success': True,
                'response_text': response_text.strip(),
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Error invoking Bedrock Agent: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    def build_curation_prompt(self, invoice_data: Dict[str, Any], textract_data: Dict[str, Any]) -> str:
        """Build the curation prompt for the Bedrock Agent"""
        return f"""
You are a specialized AI agent for invoice processing and curation. Your task is to analyze and curate the following invoice data that has been processed through OCR (Textract) and AI analysis (SageMaker).

INVOICE DATA:
{json.dumps(invoice_data, indent=2)}

ORIGINAL OCR DATA:
{json.dumps(textract_data, indent=2)}

Your tasks:
1. Review and validate all extracted fields for accuracy
2. Identify any missing critical information
3. Resolve conflicts between different extraction methods
4. Ensure data consistency and format standardization
5. Flag any suspicious or unusual patterns

Please provide your curated analysis in the following JSON format:
{{
    "curated_invoice": {{
        "invoice_id": "final_invoice_id",
        "vendor_name": "standardized_vendor_name",
        "amount": numeric_amount,
        "due_date": "YYYY-MM-DD",
        "invoice_date": "YYYY-MM-DD",
        "tax_amount": numeric_or_null,
        "subtotal": numeric_or_null,
        "currency": "USD",
        "status": "validated|flagged|error"
    }},
    "curation_notes": [
        "list of changes made during curation"
    ],
    "data_quality_score": 0.0_to_1.0,
    "validation_flags": [
        "list of any validation concerns"
    ]
}}

Respond only with valid JSON.
"""
    
    def build_insights_prompt(self, curated_data: Dict[str, Any]) -> str:
        """Build the insights generation prompt"""
        return f"""
You are an AI financial analyst specializing in invoice analysis and risk assessment. Analyze the following curated invoice data and generate actionable insights and alerts.

CURATED INVOICE DATA:
{json.dumps(curated_data, indent=2)}

Your analysis should cover:
1. Risk assessment for potential fraud or errors
2. Compliance and regulatory considerations
3. Payment timing analysis
4. Vendor relationship insights
5. Expense categorization accuracy

Please provide your analysis in the following JSON format:
{{
    "alerts": [
        {{
            "type": "duplicate|anomaly|risk|compliance",
            "severity": "low|medium|high|critical",
            "title": "Alert title",
            "description": "Detailed description",
            "recommendation": "Suggested action",
            "auto_resolvable": true_or_false
        }}
    ],
    "insights": [
        {{
            "category": "payment|vendor|amount|timing|compliance",
            "insight": "Description of the insight",
            "impact": "Potential business impact",
            "confidence": 0.0_to_1.0
        }}
    ],
    "risk_factors": [
        {{
            "factor": "Risk factor description",
            "likelihood": "low|medium|high",
            "impact": "low|medium|high",
            "mitigation": "Suggested mitigation"
        }}
    ],
    "overall_risk_level": "low|medium|high|critical"
}}

Respond only with valid JSON.
"""
    
    def build_recommendations_prompt(self, curated_data: Dict[str, Any], alerts: List[Dict[str, Any]]) -> str:
        """Build the recommendations generation prompt"""
        return f"""
You are an AI business process consultant specializing in accounts payable optimization. Based on the invoice data and identified alerts, provide actionable recommendations.

INVOICE DATA:
{json.dumps(curated_data, indent=2)}

IDENTIFIED ALERTS:
{json.dumps(alerts, indent=2)}

Please provide comprehensive recommendations in the following JSON format:
{{
    "recommendations": [
        {{
            "type": "process|payment|vendor|compliance",
            "priority": "low|medium|high|urgent",
            "title": "Recommendation title",
            "description": "Detailed recommendation",
            "expected_benefit": "Expected outcome",
            "implementation_effort": "low|medium|high"
        }}
    ],
    "next_actions": [
        {{
            "action": "Specific action to take",
            "responsible_party": "Who should do it",
            "timeline": "When it should be done",
            "automated": true_or_false
        }}
    ],
    "automation_suggestions": [
        {{
            "process": "Process that can be automated",
            "technology": "Suggested technology/tool",
            "roi_estimate": "Expected return on investment",
            "implementation_complexity": "low|medium|high"
        }}
    ],
    "summary": "Brief summary of key recommendations"
}}

Respond only with valid JSON.
"""
    
    def parse_curation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the curation response from Bedrock Agent"""
        try:
            # Try to extract JSON from the response
            import re
            
            # Look for JSON content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # If no JSON found, create a structured response
            logger.warning("Could not parse curation response as JSON")
            return {
                'parsing_error': True,
                'raw_response': response_text,
                'curated_invoice': {},
                'curation_notes': ['Failed to parse agent response'],
                'data_quality_score': 0.0,
                'validation_flags': ['Response parsing failed']
            }
            
        except Exception as e:
            logger.error(f"Error parsing curation response: {str(e)}")
            return {
                'parsing_error': True,
                'error': str(e),
                'raw_response': response_text
            }
    
    def parse_insights_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the insights response from Bedrock Agent"""
        try:
            import re
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            logger.warning("Could not parse insights response as JSON")
            return {
                'parsing_error': True,
                'raw_response': response_text,
                'alerts': [],
                'insights': [],
                'risk_factors': [],
                'overall_risk_level': 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Error parsing insights response: {str(e)}")
            return {
                'parsing_error': True,
                'error': str(e),
                'raw_response': response_text
            }
    
    def parse_recommendations_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the recommendations response from Bedrock Agent"""
        try:
            import re
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            logger.warning("Could not parse recommendations response as JSON")
            return {
                'parsing_error': True,
                'raw_response': response_text,
                'recommendations': [],
                'next_actions': [],
                'automation_suggestions': [],
                'summary': 'Failed to parse recommendations'
            }
            
        except Exception as e:
            logger.error(f"Error parsing recommendations response: {str(e)}")
            return {
                'parsing_error': True,
                'error': str(e),
                'raw_response': response_text
            }
    
    def combine_bedrock_results(self, original_data: Dict[str, Any], processing_results: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Combine all Bedrock processing results"""
        try:
            # Start with the original data structure
            combined = {
                'job_id': job_id,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'original_ai_data': original_data
            }
            
            # Add curated data
            curation_result = processing_results.get('curation', {})
            if curation_result.get('success') and not curation_result.get('curated_data', {}).get('parsing_error'):
                curated_data = curation_result['curated_data']
                combined.update({
                    'invoice_id': curated_data.get('curated_invoice', {}).get('invoice_id'),
                    'vendor_name': curated_data.get('curated_invoice', {}).get('vendor_name'),
                    'amount': curated_data.get('curated_invoice', {}).get('amount'),
                    'due_date': curated_data.get('curated_invoice', {}).get('due_date'),
                    'invoice_date': curated_data.get('curated_invoice', {}).get('invoice_date'),
                    'tax_amount': curated_data.get('curated_invoice', {}).get('tax_amount'),
                    'subtotal': curated_data.get('curated_invoice', {}).get('subtotal'),
                    'currency': curated_data.get('curated_invoice', {}).get('currency', 'USD'),
                    'status': curated_data.get('curated_invoice', {}).get('status', 'processed'),
                    'data_quality_score': curated_data.get('data_quality_score', 0.0),
                    'curation_notes': curated_data.get('curation_notes', []),
                    'validation_flags': curated_data.get('validation_flags', [])
                })
            
            # Add insights and alerts
            insights_result = processing_results.get('insights', {})
            if insights_result.get('success'):
                combined.update({
                    'alerts': insights_result.get('alerts', []),
                    'insights': insights_result.get('insights', []),
                    'risk_factors': insights_result.get('risk_factors', []),
                    'overall_risk_level': insights_result.get('overall_risk_level', 'unknown')
                })
            
            # Add recommendations
            recommendations_result = processing_results.get('recommendations', {})
            if recommendations_result.get('success'):
                combined.update({
                    'recommendations': recommendations_result.get('recommendations', []),
                    'next_actions': recommendations_result.get('next_actions', []),
                    'automation_suggestions': recommendations_result.get('automation_suggestions', []),
                    'recommendations_summary': recommendations_result.get('summary', '')
                })
            
            # Add processing metadata
            combined['bedrock_processing_results'] = processing_results
            combined['agent_session_info'] = {
                'agent_id': self.agent_id,
                'agent_alias_id': self.agent_alias_id,
                'processing_stages': len(processing_results),
                'successful_stages': len([r for r in processing_results.values() if r.get('success')])
            }
            
            # Calculate final processing score
            combined['final_processing_score'] = self.calculate_final_score(combined, processing_results)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining Bedrock results: {str(e)}")
            return {
                'job_id': job_id,
                'error': str(e),
                'original_ai_data': original_data,
                'processing_results': processing_results
            }
    
    def calculate_final_score(self, combined_data: Dict[str, Any], processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final processing quality score"""
        try:
            scores = {
                'data_quality': combined_data.get('data_quality_score', 0.0),
                'processing_success': 0.0,
                'risk_assessment': 0.0,
                'completeness': 0.0
            }
            
            # Processing success score
            successful_stages = len([r for r in processing_results.values() if r.get('success')])
            total_stages = len(processing_results)
            scores['processing_success'] = successful_stages / total_stages if total_stages > 0 else 0.0
            
            # Risk assessment score (inverse of risk level)
            risk_level = combined_data.get('overall_risk_level', 'medium')
            risk_scores = {'low': 1.0, 'medium': 0.7, 'high': 0.4, 'critical': 0.1}
            scores['risk_assessment'] = risk_scores.get(risk_level, 0.5)
            
            # Completeness score (based on required fields)
            required_fields = ['invoice_id', 'vendor_name', 'amount', 'due_date']
            completed_fields = sum(1 for field in required_fields if combined_data.get(field))
            scores['completeness'] = completed_fields / len(required_fields)
            
            # Overall score
            overall_score = sum(scores.values()) / len(scores)
            
            return {
                'individual_scores': scores,
                'overall_score': overall_score,
                'grade': self.score_to_grade(overall_score),
                'calculation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating final score: {str(e)}")
            return {'error': str(e), 'overall_score': 0.0}
    
    def score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def store_in_agent_memory(self, session_id: str, invoice_data: Dict[str, Any]) -> None:
        """Store processed invoice data in agent memory (DynamoDB)"""
        try:
            if not self.memory_table:
                logger.warning("Memory table not available, skipping memory storage")
                return
            
            # Prepare memory record
            memory_record = {
                'session_id': session_id,
                'job_id': invoice_data.get('job_id'),
                'invoice_id': invoice_data.get('invoice_id', 'unknown'),
                'vendor_name': invoice_data.get('vendor_name', 'unknown'),
                'amount': invoice_data.get('amount', 0),
                'due_date': invoice_data.get('due_date'),
                'processing_timestamp': datetime.utcnow().isoformat(),
                'ttl': int((datetime.utcnow() + timedelta(hours=self.session_ttl_hours)).timestamp()),
                'alerts_count': len(invoice_data.get('alerts', [])),
                'risk_level': invoice_data.get('overall_risk_level', 'unknown'),
                'data_quality_score': invoice_data.get('data_quality_score', 0.0),
                'status': invoice_data.get('status', 'processed')
            }
            
            # Store in DynamoDB
            self.memory_table.put_item(Item=memory_record)
            logger.info(f"Stored invoice data in agent memory: {session_id}")
            
        except Exception as e:
            logger.error(f"Error storing in agent memory: {str(e)}")
            # Don't raise - memory storage is not critical
    
    def store_bedrock_results(self, bucket: str, key: str, results: Dict[str, Any]) -> None:
        """Store Bedrock processing results to S3"""
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(results, default=str, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Stored Bedrock results: s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"Error storing Bedrock results: {str(e)}")
            # Don't raise - this is not critical

# Initialize the processor
bedrock_processor = PayScanBedrockProcessor()

def lambda_handler(event, context):
    """Lambda entry point"""
    return bedrock_processor.lambda_handler(event, context)