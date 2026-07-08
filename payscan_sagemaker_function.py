"""
PayScan AI Agent - SageMaker Processing Function
Handles AI processing using Mistral-7B model hosted on SageMaker
"""

import json
import boto3
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')

class PayScanSageMakerProcessor:
    def __init__(self):
        self.endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT_NAME', 'payscan-mistral-endpoint')
        self.max_tokens = 1000
        self.temperature = 0.1  # Low temperature for consistent extraction
        
        # Prompt templates for different analysis tasks
        self.prompts = {
            'extract_and_validate': """
You are an AI assistant specialized in invoice processing. Analyze the following invoice data extracted by OCR and provide a structured, validated response.

OCR Extracted Data:
{textract_data}

Your tasks:
1. Validate and correct any OCR errors in the key fields
2. Extract missing information if possible
3. Identify any anomalies or inconsistencies
4. Provide confidence scores for each field

Return your response as a valid JSON object with this exact structure:
{{
    "validated_fields": {{
        "invoice_id": "extracted_invoice_id",
        "vendor": "vendor_name",
        "amount": numeric_amount,
        "due_date": "YYYY-MM-DD",
        "invoice_date": "YYYY-MM-DD",
        "tax_amount": numeric_tax_amount_or_null,
        "subtotal": numeric_subtotal_or_null
    }},
    "confidence_scores": {{
        "invoice_id": 0.95,
        "vendor": 0.98,
        "amount": 0.92,
        "due_date": 0.87
    }},
    "anomalies": [
        "list of any detected anomalies or inconsistencies"
    ],
    "ocr_corrections": [
        "list of corrections made to OCR data"
    ],
    "extraction_notes": "any additional notes about the extraction process"
}}

Respond ONLY with valid JSON. Do not include any other text or explanations.
""",
            
            'analyze_discrepancies': """
You are an AI financial analyst. Analyze this invoice data for potential discrepancies, fraud indicators, or anomalies.

Invoice Data:
{invoice_data}

Analyze for:
1. Mathematical inconsistencies (subtotal + tax ≠ total)
2. Unusual patterns or amounts
3. Potential duplicate indicators
4. Date inconsistencies
5. Vendor name variations

Return your response as a valid JSON object:
{{
    "risk_assessment": {{
        "overall_risk_score": 0.0_to_1.0,
        "risk_factors": ["list_of_identified_risks"]
    }},
    "discrepancies": [
        {{
            "type": "mathematical|date|vendor|other",
            "description": "description_of_discrepancy",
            "severity": "low|medium|high",
            "recommendation": "recommended_action"
        }}
    ],
    "validation_status": "approved|flagged|rejected",
    "analyst_notes": "additional_analysis_notes"
}}

Respond ONLY with valid JSON.
""",
            
            'categorize_and_summarize': """
You are an AI accounting assistant. Categorize and summarize this invoice for financial reporting.

Invoice Data:
{invoice_data}

Tasks:
1. Categorize the expense type
2. Extract line items if present
3. Identify tax jurisdictions
4. Suggest account codes

Return your response as a valid JSON object:
{{
    "categorization": {{
        "expense_category": "office_supplies|utilities|services|equipment|other",
        "subcategory": "specific_subcategory",
        "account_code_suggestion": "suggested_GL_code",
        "tax_jurisdiction": "detected_tax_region"
    }},
    "line_items": [
        {{
            "description": "item_description",
            "quantity": numeric_quantity,
            "unit_price": numeric_price,
            "total": numeric_total
        }}
    ],
    "summary": {{
        "vendor_summary": "brief_vendor_description",
        "expense_summary": "brief_expense_description",
        "payment_terms": "detected_payment_terms"
    }}
}}

Respond ONLY with valid JSON.
"""
        }
    
    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Main Lambda handler for SageMaker processing"""
        try:
            logger.info(f"Starting SageMaker processing: {json.dumps(event, default=str)}")
            
            textract_data = event['textract_data']
            job_id = event['job_id']
            processing_bucket = event['processing_bucket']
            endpoint_name = event.get('sagemaker_endpoint', self.endpoint_name)
            
            # Validate that we have the necessary data
            if not textract_data.get('success'):
                raise Exception("Textract processing was not successful")
            
            # Extract the structured data from Textract
            structured_data = textract_data.get('structured_data', {})
            
            # Process with SageMaker in stages
            processing_results = {}
            
            # Stage 1: Extract and validate fields
            logger.info("Stage 1: Extracting and validating fields")
            validation_result = self.process_with_mistral(
                endpoint_name,
                self.prompts['extract_and_validate'],
                {'textract_data': json.dumps(structured_data, indent=2)}
            )
            processing_results['validation'] = validation_result
            
            # Stage 2: Analyze for discrepancies
            logger.info("Stage 2: Analyzing for discrepancies")
            if validation_result.get('success'):
                discrepancy_result = self.process_with_mistral(
                    endpoint_name,
                    self.prompts['analyze_discrepancies'],
                    {'invoice_data': json.dumps(validation_result['parsed_response'], indent=2)}
                )
                processing_results['discrepancy_analysis'] = discrepancy_result
            
            # Stage 3: Categorize and summarize
            logger.info("Stage 3: Categorizing and summarizing")
            if validation_result.get('success'):
                categorization_result = self.process_with_mistral(
                    endpoint_name,
                    self.prompts['categorize_and_summarize'],
                    {'invoice_data': json.dumps(validation_result['parsed_response'], indent=2)}
                )
                processing_results['categorization'] = categorization_result
            
            # Combine all results into final structure
            final_result = self.combine_processing_results(
                structured_data,
                processing_results
            )
            
            # Store results
            storage_key = f"sagemaker/{job_id}/ai_analysis.json"
            self.store_analysis_results(processing_bucket, storage_key, final_result)
            
            return {
                'success': True,
                'job_id': job_id,
                'endpoint_used': endpoint_name,
                'analysis_results_s3': f"s3://{processing_bucket}/{storage_key}",
                'structured_invoice_data': final_result,
                'processing_stats': {
                    'stages_completed': len([r for r in processing_results.values() if r.get('success')]),
                    'total_stages': len(processing_results),
                    'overall_confidence': self.calculate_overall_confidence(processing_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in SageMaker processing: {str(e)}")
            return {
                'success': False,
                'job_id': event.get('job_id'),
                'error': str(e),
                'error_type': 'sagemaker_processing_error'
            }
    
    def process_with_mistral(self, endpoint_name: str, prompt_template: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Process a prompt with the Mistral model on SageMaker"""
        try:
            # Format the prompt with variables
            formatted_prompt = prompt_template.format(**variables)
            
            # Prepare the request payload for Mistral
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "do_sample": False,
                    "stop": ["</s>", "\n\nHuman:", "\n\nAssistant:"]
                }
            }
            
            logger.info(f"Invoking SageMaker endpoint: {endpoint_name}")
            
            # Invoke the SageMaker endpoint
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            # Parse the response
            response_body = response['Body'].read().decode('utf-8')
            model_response = json.loads(response_body)
            
            # Extract the generated text
            generated_text = ""
            if isinstance(model_response, list) and len(model_response) > 0:
                generated_text = model_response[0].get('generated_text', '')
            elif isinstance(model_response, dict):
                generated_text = model_response.get('generated_text', '') or model_response.get('outputs', '')
            
            # Clean the response (remove the original prompt)
            if formatted_prompt in generated_text:
                generated_text = generated_text.replace(formatted_prompt, '').strip()
            
            logger.info(f"Received response from Mistral: {len(generated_text)} characters")
            
            # Parse JSON response
            parsed_response = self.parse_json_response(generated_text)
            
            return {
                'success': True,
                'raw_response': generated_text,
                'parsed_response': parsed_response,
                'model_used': 'mistral-7b-instruct',
                'endpoint': endpoint_name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing with Mistral: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'mistral_processing_error'
            }
    
    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from model, handling common formatting issues"""
        try:
            # Try direct JSON parsing first
            return json.loads(response_text.strip())
            
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                try:
                    return json.loads(json_match.group(1).strip())
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON-like content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, return a structured error response
            logger.warning(f"Could not parse JSON response: {response_text[:200]}...")
            return {
                'parsing_error': True,
                'raw_text': response_text,
                'error_message': 'Could not parse model response as JSON'
            }
            
    def combine_processing_results(self, 
                                  original_data: Dict[str, Any], 
                                  processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all processing results into final structured data"""
        try:
            # Start with original Textract data
            combined = {
                'original_textract_data': original_data,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
            # Add validated fields from Stage 1
            if processing_results.get('validation', {}).get('success'):
                validation_data = processing_results['validation']['parsed_response']
                if not validation_data.get('parsing_error'):
                    combined.update({
                        'invoice_id': validation_data.get('validated_fields', {}).get('invoice_id'),
                        'vendor': validation_data.get('validated_fields', {}).get('vendor'),
                        'amount': validation_data.get('validated_fields', {}).get('amount'),
                        'due_date': validation_data.get('validated_fields', {}).get('due_date'),
                        'invoice_date': validation_data.get('validated_fields', {}).get('invoice_date'),
                        'tax_amount': validation_data.get('validated_fields', {}).get('tax_amount'),
                        'subtotal': validation_data.get('validated_fields', {}).get('subtotal'),
                        'confidence_scores': validation_data.get('confidence_scores', {}),
                        'ocr_corrections': validation_data.get('ocr_corrections', []),
                        'extraction_notes': validation_data.get('extraction_notes', '')
                    })
            
            # Add risk assessment from Stage 2
            if processing_results.get('discrepancy_analysis', {}).get('success'):
                discrepancy_data = processing_results['discrepancy_analysis']['parsed_response']
                if not discrepancy_data.get('parsing_error'):
                    combined.update({
                        'risk_assessment': discrepancy_data.get('risk_assessment', {}),
                        'discrepancies': discrepancy_data.get('discrepancies', []),
                        'validation_status': discrepancy_data.get('validation_status', 'unknown'),
                        'analyst_notes': discrepancy_data.get('analyst_notes', '')
                    })
            
            # Add categorization from Stage 3
            if processing_results.get('categorization', {}).get('success'):
                categorization_data = processing_results['categorization']['parsed_response']
                if not categorization_data.get('parsing_error'):
                    combined.update({
                        'categorization': categorization_data.get('categorization', {}),
                        'line_items': categorization_data.get('line_items', []),
                        'summary': categorization_data.get('summary', {})
                    })
            
            # Add processing metadata
            combined['ai_processing_results'] = processing_results
            combined['processing_quality'] = self.assess_processing_quality(processing_results)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining processing results: {str(e)}")
            return {
                'error': str(e),
                'original_textract_data': original_data,
                'processing_results': processing_results
            }
    
    def assess_processing_quality(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of AI processing results"""
        try:
            quality_metrics = {
                'stages_successful': 0,
                'total_stages': len(processing_results),
                'parsing_errors': 0,
                'overall_quality': 'unknown'
            }
            
            for stage_name, stage_result in processing_results.items():
                if stage_result.get('success'):
                    quality_metrics['stages_successful'] += 1
                    
                    # Check for parsing errors in the response
                    if stage_result.get('parsed_response', {}).get('parsing_error'):
                        quality_metrics['parsing_errors'] += 1
            
            # Calculate overall quality score
            success_rate = quality_metrics['stages_successful'] / quality_metrics['total_stages']
            parsing_success_rate = 1.0 - (quality_metrics['parsing_errors'] / quality_metrics['total_stages'])
            
            overall_score = (success_rate + parsing_success_rate) / 2
            
            if overall_score >= 0.9:
                quality_metrics['overall_quality'] = 'excellent'
            elif overall_score >= 0.7:
                quality_metrics['overall_quality'] = 'good'
            elif overall_score >= 0.5:
                quality_metrics['overall_quality'] = 'fair'
            else:
                quality_metrics['overall_quality'] = 'poor'
            
            quality_metrics['quality_score'] = overall_score
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing processing quality: {str(e)}")
            return {'error': str(e)}
    
    def calculate_overall_confidence(self, processing_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score from all processing stages"""
        try:
            confidence_scores = []
            
            # Get confidence scores from validation stage
            validation_result = processing_results.get('validation', {})
            if validation_result.get('success'):
                parsed_response = validation_result.get('parsed_response', {})
                if not parsed_response.get('parsing_error'):
                    stage_confidences = parsed_response.get('confidence_scores', {})
                    confidence_scores.extend(stage_confidences.values())
            
            # Calculate average confidence
            if confidence_scores:
                return sum(confidence_scores) / len(confidence_scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {str(e)}")
            return 0.0
    
    def store_analysis_results(self, bucket: str, key: str, results: Dict[str, Any]) -> None:
        """Store AI analysis results to S3"""
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(results, default=str, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Stored AI analysis results: s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"Error storing analysis results: {str(e)}")
            # Don't raise - this is not critical for processing

# Initialize the processor
sagemaker_processor = PayScanSageMakerProcessor()

def lambda_handler(event, context):
    """Lambda entry point"""
    return sagemaker_processor.lambda_handler(event, context)