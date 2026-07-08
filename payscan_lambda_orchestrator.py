"""
PayScan AI Agent - Main Orchestrator Lambda Function
Handles S3 uploads, manages SageMaker endpoints, and triggers Step Functions
"""

import json
import boto3
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker')
stepfunctions_client = boto3.client('stepfunctions')
textract_client = boto3.client('textract')
bedrock_client = boto3.client('bedrock-agent')
sns_client = boto3.client('sns')

# Environment variables
SAGEMAKER_ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT_NAME', 'payscan-mistral-endpoint')
SAGEMAKER_ENDPOINT_CONFIG = os.environ.get('SAGEMAKER_ENDPOINT_CONFIG', 'payscan-mistral-config')
SAGEMAKER_MODEL_NAME = os.environ.get('SAGEMAKER_MODEL_NAME', 'payscan-mistral-model')
STEP_FUNCTION_ARN = os.environ.get('STEP_FUNCTION_ARN')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')
PROCESSING_BUCKET = os.environ.get('PROCESSING_BUCKET', 'payscan-processing')
RESULTS_BUCKET = os.environ.get('RESULTS_BUCKET', 'payscan-results')

class PayScanOrchestrator:
    def __init__(self):
        self.endpoint_creation_timeout = 600  # 10 minutes
        self.supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
        self.max_file_size = 10 * 1024 * 1024  # 10MB
    
    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Main Lambda handler for PayScan AI Agent
        Supports both S3 events and API Gateway calls
        """
        try:
            logger.info(f"Received event: {json.dumps(event, default=str)}")
            
            # Determine event source
            if 'Records' in event and event['Records'][0].get('eventSource') == 'aws:s3':
                # S3 event - file upload
                return self.handle_s3_upload(event)
            elif 'httpMethod' in event:
                # API Gateway event
                return self.handle_api_request(event)
            else:
                # Direct invocation
                return self.handle_direct_invocation(event)
                
        except Exception as e:
            logger.error(f"Error in lambda_handler: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Internal server error',
                    'message': str(e)
                })
            }
    
    def handle_s3_upload(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle S3 upload events"""
        try:
            results = []
            
            for record in event['Records']:
                bucket = record['s3']['bucket']['name']
                key = record['s3']['object']['key']
                
                logger.info(f"Processing S3 upload: s3://{bucket}/{key}")
                
                # Validate file
                if not self.validate_uploaded_file(bucket, key):
                    logger.warning(f"File validation failed for {key}")
                    continue
                
                # Start processing pipeline
                job_id = str(uuid.uuid4())
                result = self.start_processing_pipeline(bucket, key, job_id)
                results.append(result)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Processing started',
                    'results': results
                })
            }
            
        except Exception as e:
            logger.error(f"Error handling S3 upload: {str(e)}")
            raise
    
    def handle_api_request(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API Gateway requests"""
        try:
            method = event['httpMethod']
            path = event['path']
            
            # Add CORS headers
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
                'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
            }
            
            if method == 'OPTIONS':
                return {'statusCode': 200, 'headers': headers, 'body': ''}
            
            # Route API requests
            if path.startswith('/upload') and method == 'POST':
                return self.handle_upload_api(event, headers)
            elif path.startswith('/process') and method == 'POST':
                return self.handle_process_api(event, headers)
            elif path.startswith('/status') and method == 'GET':
                return self.handle_status_api(event, headers)
            elif path.startswith('/invoices') and method == 'GET':
                return self.handle_invoices_api(event, headers)
            else:
                return {
                    'statusCode': 404,
                    'headers': headers,
                    'body': json.dumps({'error': 'Not found'})
                }
                
        except Exception as e:
            logger.error(f"Error handling API request: {str(e)}")
            return {
                'statusCode': 500,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': str(e)})
            }
    
    def handle_direct_invocation(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle direct Lambda invocations"""
        action = event.get('action')
        
        if action == 'start_endpoint':
            return self.start_sagemaker_endpoint()
        elif action == 'stop_endpoint':
            return self.stop_sagemaker_endpoint()
        elif action == 'check_endpoint':
            return self.check_endpoint_status()
        elif action == 'process_batch':
            return self.process_invoice_batch(event.get('batch_data', []))
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Unknown action'})
            }
    
    def validate_uploaded_file(self, bucket: str, key: str) -> bool:
        """Validate uploaded file"""
        try:
            # Check file extension
            file_extension = os.path.splitext(key)[1].lower()
            if file_extension not in self.supported_extensions:
                logger.warning(f"Unsupported file extension: {file_extension}")
                return False
            
            # Check file size
            response = s3_client.head_object(Bucket=bucket, Key=key)
            file_size = response['ContentLength']
            
            if file_size > self.max_file_size:
                logger.warning(f"File too large: {file_size} bytes")
                return False
            
            # Check if file is accessible
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
            except Exception as e:
                logger.error(f"File not accessible: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file: {str(e)}")
            return False
    
    def start_processing_pipeline(self, bucket: str, key: str, job_id: str) -> Dict[str, Any]:
        """Start the complete processing pipeline"""
        try:
            # Create processing job metadata
            job_metadata = {
                'job_id': job_id,
                'source_bucket': bucket,
                'source_key': key,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'started',
                'steps': []
            }
            
            # Step 1: Ensure SageMaker endpoint is running
            endpoint_result = self.ensure_sagemaker_endpoint()
            job_metadata['steps'].append({
                'step': 'sagemaker_endpoint',
                'result': endpoint_result,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Step 2: Start Step Functions execution
            step_function_input = {
                'job_id': job_id,
                'source_bucket': bucket,
                'source_key': key,
                'sagemaker_endpoint': SAGEMAKER_ENDPOINT_NAME,
                'processing_bucket': PROCESSING_BUCKET,
                'results_bucket': RESULTS_BUCKET
            }
            
            response = stepfunctions_client.start_execution(
                stateMachineArn=STEP_FUNCTION_ARN,
                name=f"payscan-{job_id}",
                input=json.dumps(step_function_input)
            )
            
            job_metadata['steps'].append({
                'step': 'step_functions',
                'execution_arn': response['executionArn'],
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Store job metadata
            self.store_job_metadata(job_id, job_metadata)
            
            return {
                'job_id': job_id,
                'execution_arn': response['executionArn'],
                'status': 'started'
            }
            
        except Exception as e:
            logger.error(f"Error starting processing pipeline: {str(e)}")
            raise
    
    def ensure_sagemaker_endpoint(self) -> Dict[str, Any]:
        """Ensure SageMaker endpoint is running"""
        try:
            # Check current endpoint status
            status = self.check_endpoint_status()
            
            if status['status'] == 'InService':
                logger.info("SageMaker endpoint already running")
                return status
            elif status['status'] == 'Creating':
                logger.info("SageMaker endpoint is being created")
                return status
            else:
                # Start the endpoint
                logger.info("Starting SageMaker endpoint")
                return self.start_sagemaker_endpoint()
                
        except Exception as e:
            logger.error(f"Error ensuring SageMaker endpoint: {str(e)}")
            raise
    
    def check_endpoint_status(self) -> Dict[str, Any]:
        """Check SageMaker endpoint status"""
        try:
            response = sagemaker_client.describe_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT_NAME
            )
            
            return {
                'endpoint_name': SAGEMAKER_ENDPOINT_NAME,
                'status': response['EndpointStatus'],
                'creation_time': response.get('CreationTime'),
                'last_modified_time': response.get('LastModifiedTime')
            }
            
        except sagemaker_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                return {
                    'endpoint_name': SAGEMAKER_ENDPOINT_NAME,
                    'status': 'NotFound',
                    'message': 'Endpoint does not exist'
                }
            else:
                raise
    
    def start_sagemaker_endpoint(self) -> Dict[str, Any]:
        """Start SageMaker endpoint"""
        try:
            # Check if endpoint exists
            status = self.check_endpoint_status()
            
            if status['status'] == 'InService':
                return status
            elif status['status'] == 'Creating':
                return status
            
            # Create endpoint if it doesn't exist
            if status['status'] == 'NotFound':
                logger.info(f"Creating SageMaker endpoint: {SAGEMAKER_ENDPOINT_NAME}")
                
                response = sagemaker_client.create_endpoint(
                    EndpointName=SAGEMAKER_ENDPOINT_NAME,
                    EndpointConfigName=SAGEMAKER_ENDPOINT_CONFIG,
                    Tags=[
                        {'Key': 'Project', 'Value': 'PayScan'},
                        {'Key': 'Environment', 'Value': 'Production'},
                        {'Key': 'AutoStop', 'Value': 'true'}
                    ]
                )
                
                return {
                    'endpoint_name': SAGEMAKER_ENDPOINT_NAME,
                    'status': 'Creating',
                    'endpoint_arn': response['EndpointArn'],
                    'message': 'Endpoint creation started'
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error starting SageMaker endpoint: {str(e)}")
            raise
    
    def stop_sagemaker_endpoint(self) -> Dict[str, Any]:
        """Stop SageMaker endpoint to save costs"""
        try:
            logger.info(f"Stopping SageMaker endpoint: {SAGEMAKER_ENDPOINT_NAME}")
            
            sagemaker_client.delete_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT_NAME
            )
            
            return {
                'endpoint_name': SAGEMAKER_ENDPOINT_NAME,
                'status': 'Deleting',
                'message': 'Endpoint deletion started'
            }
            
        except sagemaker_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                return {
                    'endpoint_name': SAGEMAKER_ENDPOINT_NAME,
                    'status': 'NotFound',
                    'message': 'Endpoint does not exist'
                }
            else:
                raise
    
    def store_job_metadata(self, job_id: str, metadata: Dict[str, Any]) -> None:
        """Store job metadata in S3"""
        try:
            s3_client.put_object(
                Bucket=PROCESSING_BUCKET,
                Key=f"jobs/{job_id}/metadata.json",
                Body=json.dumps(metadata, default=str),
                ContentType='application/json'
            )
            
        except Exception as e:
            logger.error(f"Error storing job metadata: {str(e)}")
            # Don't raise - this is not critical for processing
    
    def handle_upload_api(self, event: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Handle file upload via API"""
        try:
            # Extract file data from event
            body = event.get('body', '')
            if event.get('isBase64Encoded'):
                import base64
                body = base64.b64decode(body)
            
            # Generate unique filename
            file_id = str(uuid.uuid4())
            filename = f"uploads/{file_id}.pdf"  # Assume PDF for API uploads
            
            # Upload to S3
            s3_client.put_object(
                Bucket=PROCESSING_BUCKET,
                Key=filename,
                Body=body,
                ContentType='application/pdf'
            )
            
            # Start processing
            job_id = str(uuid.uuid4())
            result = self.start_processing_pipeline(PROCESSING_BUCKET, filename, job_id)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'message': 'File uploaded successfully',
                    'file_id': file_id,
                    'job_id': job_id,
                    'processing_started': True
                })
            }
            
        except Exception as e:
            logger.error(f"Error in upload API: {str(e)}")
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': str(e)})
            }
    
    def handle_process_api(self, event: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Handle process request via API"""
        try:
            body = json.loads(event.get('body', '{}'))
            file_key = body.get('file_key')
            bucket = body.get('bucket', PROCESSING_BUCKET)
            
            if not file_key:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({'error': 'file_key is required'})
                }
            
            job_id = str(uuid.uuid4())
            result = self.start_processing_pipeline(bucket, file_key, job_id)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(result)
            }
            
        except Exception as e:
            logger.error(f"Error in process API: {str(e)}")
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': str(e)})
            }
    
    def handle_status_api(self, event: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Handle status check via API"""
        try:
            job_id = event.get('pathParameters', {}).get('job_id')
            
            if job_id:
                # Get specific job status
                status = self.get_job_status(job_id)
            else:
                # Get general system status
                status = {
                    'system_status': 'operational',
                    'sagemaker_endpoint': self.check_endpoint_status(),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(status, default=str)
            }
            
        except Exception as e:
            logger.error(f"Error in status API: {str(e)}")
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': str(e)})
            }
    
    def handle_invoices_api(self, event: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Handle invoices listing via API"""
        try:
            # Get query parameters for filtering
            query_params = event.get('queryStringParameters') or {}
            
            # List processed invoices from results bucket
            invoices = self.list_processed_invoices(query_params)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'invoices': invoices,
                    'count': len(invoices)
                }, default=str)
            }
            
        except Exception as e:
            logger.error(f"Error in invoices API: {str(e)}")
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': str(e)})
            }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job processing status"""
        try:
            # Get job metadata
            response = s3_client.get_object(
                Bucket=PROCESSING_BUCKET,
                Key=f"jobs/{job_id}/metadata.json"
            )
            metadata = json.loads(response['Body'].read())
            
            # Check Step Functions execution status if available
            for step in metadata.get('steps', []):
                if step.get('step') == 'step_functions' and 'execution_arn' in step:
                    try:
                        sf_response = stepfunctions_client.describe_execution(
                            executionArn=step['execution_arn']
                        )
                        metadata['step_functions_status'] = sf_response['status']
                        metadata['step_functions_output'] = sf_response.get('output')
                    except Exception as e:
                        logger.error(f"Error checking Step Functions status: {str(e)}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return {
                'job_id': job_id,
                'status': 'unknown',
                'error': str(e)
            }
    
    def list_processed_invoices(self, filters: Dict[str, str]) -> list:
        """List processed invoices with optional filtering"""
        try:
            invoices = []
            
            # List objects in results bucket
            response = s3_client.list_objects_v2(
                Bucket=RESULTS_BUCKET,
                Prefix='processed/'
            )
            
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    try:
                        # Get invoice data
                        invoice_response = s3_client.get_object(
                            Bucket=RESULTS_BUCKET,
                            Key=obj['Key']
                        )
                        invoice_data = json.loads(invoice_response['Body'].read())
                        
                        # Apply filters
                        if self.matches_filters(invoice_data, filters):
                            invoices.append(invoice_data)
                            
                    except Exception as e:
                        logger.error(f"Error reading invoice {obj['Key']}: {str(e)}")
                        continue
            
            return sorted(invoices, key=lambda x: x.get('processed_date', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing invoices: {str(e)}")
            return []
    
    def matches_filters(self, invoice_data: Dict[str, Any], filters: Dict[str, str]) -> bool:
        """Check if invoice matches the given filters"""
        try:
            # Status filter
            if filters.get('status') and invoice_data.get('status') != filters['status']:
                return False
            
            # Date range filter
            if filters.get('date_from') or filters.get('date_to'):
                invoice_date = invoice_data.get('processed_date', '')
                if filters.get('date_from') and invoice_date < filters['date_from']:
                    return False
                if filters.get('date_to') and invoice_date > filters['date_to']:
                    return False
            
            # Search filter
            if filters.get('search'):
                search_term = filters['search'].lower()
                searchable_text = f"{invoice_data.get('vendor', '')} {invoice_data.get('invoice_id', '')}".lower()
                if search_term not in searchable_text:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return True  # Include on error

# Initialize the orchestrator
orchestrator = PayScanOrchestrator()

def lambda_handler(event, context):
    """Lambda entry point"""
    return orchestrator.lambda_handler(event, context)