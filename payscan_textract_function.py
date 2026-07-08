"""
PayScan AI Agent - Textract Processing Function
Handles OCR extraction from invoices using Amazon Textract
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
textract_client = boto3.client('textract')
s3_client = boto3.client('s3')

class PayScanTextractProcessor:
    def __init__(self):
        self.confidence_threshold = 90.0
        self.key_value_confidence_threshold = 85.0
        
        # Define invoice field patterns
        self.invoice_patterns = {
            'invoice_id': [
                r'invoice\s*#?\s*:?\s*([A-Za-z0-9\-]+)',
                r'inv\s*#?\s*:?\s*([A-Za-z0-9\-]+)',
                r'bill\s*#?\s*:?\s*([A-Za-z0-9\-]+)',
                r'reference\s*#?\s*:?\s*([A-Za-z0-9\-]+)'
            ],
            'amount': [
                r'total\s*:?\s*\$?(\d+[\.,]\d{2})',
                r'amount\s*due\s*:?\s*\$?(\d+[\.,]\d{2})',
                r'balance\s*:?\s*\$?(\d+[\.,]\d{2})',
                r'\$(\d+[\.,]\d{2})'
            ],
            'due_date': [
                r'due\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
                r'payment\s*due\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
                r'date\s*due\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
            ],
            'vendor': [
                r'from\s*:?\s*([A-Za-z\s]+)',
                r'bill\s*from\s*:?\s*([A-Za-z\s]+)',
                r'vendor\s*:?\s*([A-Za-z\s]+)'
            ]
        }
    
    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Main Lambda handler for Textract processing"""
        try:
            logger.info(f"Starting Textract processing: {json.dumps(event, default=str)}")
            
            source_bucket = event['source_bucket']
            source_key = event['source_key']
            job_id = event['job_id']
            processing_bucket = event['processing_bucket']
            
            # Process document with Textract
            textract_result = self.process_document(source_bucket, source_key)
            
            # Extract structured data
            structured_data = self.extract_invoice_data(textract_result)
            
            # Store intermediate results
            storage_key = f"textract/{job_id}/raw_output.json"
            self.store_textract_output(processing_bucket, storage_key, textract_result)
            
            # Store structured data
            structured_key = f"textract/{job_id}/structured_data.json"
            self.store_structured_data(processing_bucket, structured_key, structured_data)
            
            return {
                'success': True,
                'job_id': job_id,
                'textract_job_id': textract_result.get('JobId'),
                'raw_output_s3': f"s3://{processing_bucket}/{storage_key}",
                'structured_data_s3': f"s3://{processing_bucket}/{structured_key}",
                'structured_data': structured_data,
                'processing_stats': {
                    'total_blocks': len(textract_result.get('Blocks', [])),
                    'confidence_score': self.calculate_average_confidence(textract_result),
                    'pages_processed': 1
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Textract processing: {str(e)}")
            return {
                'success': False,
                'job_id': event.get('job_id'),
                'error': str(e),
                'error_type': 'textract_processing_error'
            }
    
    def process_document(self, bucket: str, key: str) -> Dict[str, Any]:
        """Process document with Textract"""
        try:
            logger.info(f"Processing document: s3://{bucket}/{key}")
            
            # Determine file type
            file_extension = os.path.splitext(key)[1].lower()
            
            if file_extension == '.pdf':
                # For PDF files, use synchronous processing for single page
                # or asynchronous for multi-page
                return self.process_pdf_document(bucket, key)
            else:
                # For image files, use synchronous processing
                return self.process_image_document(bucket, key)
                
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    def process_image_document(self, bucket: str, key: str) -> Dict[str, Any]:
        """Process image document synchronously"""
        try:
            response = textract_client.analyze_document(
                Document={
                    'S3Object': {
                        'Bucket': bucket,
                        'Name': key
                    }
                },
                FeatureTypes=['FORMS', 'TABLES']
            )
            
            logger.info(f"Textract analysis complete for image: {len(response.get('Blocks', []))} blocks found")
            return response
            
        except Exception as e:
            logger.error(f"Error processing image with Textract: {str(e)}")
            raise
    
    def process_pdf_document(self, bucket: str, key: str) -> Dict[str, Any]:
        """Process PDF document (synchronous for single page)"""
        try:
            # First try synchronous processing (works for single page PDFs)
            try:
                response = textract_client.analyze_document(
                    Document={
                        'S3Object': {
                            'Bucket': bucket,
                            'Name': key
                        }
                    },
                    FeatureTypes=['FORMS', 'TABLES']
                )
                
                logger.info(f"Textract analysis complete for PDF: {len(response.get('Blocks', []))} blocks found")
                return response
                
            except textract_client.exceptions.UnsupportedDocumentException:
                # Multi-page PDF - use asynchronous processing
                logger.info("Multi-page PDF detected, using asynchronous processing")
                return self.process_pdf_async(bucket, key)
                
        except Exception as e:
            logger.error(f"Error processing PDF with Textract: {str(e)}")
            raise
    
    def process_pdf_async(self, bucket: str, key: str) -> Dict[str, Any]:
        """Process multi-page PDF asynchronously"""
        try:
            # Start asynchronous job
            response = textract_client.start_document_analysis(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': bucket,
                        'Name': key
                    }
                },
                FeatureTypes=['FORMS', 'TABLES']
            )
            
            job_id = response['JobId']
            logger.info(f"Started async Textract job: {job_id}")
            
            # Wait for completion (with timeout)
            import time
            max_wait_time = 300  # 5 minutes
            wait_interval = 10   # 10 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                status_response = textract_client.get_document_analysis(JobId=job_id)
                status = status_response['JobStatus']
                
                if status == 'SUCCEEDED':
                    logger.info(f"Async Textract job completed: {job_id}")
                    return self.get_async_results(job_id)
                elif status == 'FAILED':
                    raise Exception(f"Textract job failed: {status_response.get('StatusMessage', 'Unknown error')}")
                
                time.sleep(wait_interval)
                elapsed_time += wait_interval
                logger.info(f"Waiting for Textract job {job_id}... Status: {status}")
            
            raise Exception(f"Textract job timed out after {max_wait_time} seconds")
            
        except Exception as e:
            logger.error(f"Error in async PDF processing: {str(e)}")
            raise
    
    def get_async_results(self, job_id: str) -> Dict[str, Any]:
        """Get results from async Textract job"""
        try:
            all_blocks = []
            next_token = None
            
            while True:
                if next_token:
                    response = textract_client.get_document_analysis(
                        JobId=job_id,
                        NextToken=next_token
                    )
                else:
                    response = textract_client.get_document_analysis(JobId=job_id)
                
                all_blocks.extend(response.get('Blocks', []))
                
                next_token = response.get('NextToken')
                if not next_token:
                    break
            
            # Reconstruct the response format
            return {
                'JobId': job_id,
                'JobStatus': 'SUCCEEDED',
                'Blocks': all_blocks,
                'DetectDocumentTextModelVersion': response.get('DetectDocumentTextModelVersion'),
                'AnalyzeDocumentModelVersion': response.get('AnalyzeDocumentModelVersion')
            }
            
        except Exception as e:
            logger.error(f"Error getting async results: {str(e)}")
            raise
    
    def extract_invoice_data(self, textract_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured invoice data from Textract results"""
        try:
            blocks = textract_result.get('Blocks', [])
            
            # Extract text from all blocks
            extracted_text = self.extract_text_from_blocks(blocks)
            
            # Extract key-value pairs
            key_value_pairs = self.extract_key_value_pairs(blocks)
            
            # Extract table data
            tables = self.extract_tables(blocks)
            
            # Use pattern matching to find invoice fields
            invoice_fields = self.extract_invoice_fields_with_patterns(extracted_text)
            
            # Merge key-value pairs with pattern-matched fields
            merged_fields = self.merge_extracted_fields(invoice_fields, key_value_pairs)
            
            # Calculate confidence scores
            confidence_scores = self.calculate_field_confidence(blocks, merged_fields)
            
            structured_data = {
                'invoice_id': merged_fields.get('invoice_id'),
                'vendor': merged_fields.get('vendor'),
                'amount': self.parse_amount(merged_fields.get('amount')),
                'due_date': self.parse_date(merged_fields.get('due_date')),
                'raw_text': extracted_text,
                'key_value_pairs': key_value_pairs,
                'tables': tables,
                'confidence_scores': confidence_scores,
                'extraction_metadata': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'textract_job_id': textract_result.get('JobId'),
                    'total_blocks': len(blocks),
                    'method': 'pattern_matching_and_forms'
                }
            }
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error extracting invoice data: {str(e)}")
            raise
    
    def extract_text_from_blocks(self, blocks: List[Dict[str, Any]]) -> str:
        """Extract all text from Textract blocks"""
        text_blocks = []
        
        for block in blocks:
            if block.get('BlockType') == 'LINE':
                text = block.get('Text', '').strip()
                if text:
                    text_blocks.append(text)
        
        return '\n'.join(text_blocks)
    
    def extract_key_value_pairs(self, blocks: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract key-value pairs from Textract forms analysis"""
        key_value_pairs = {}
        
        # Create a map of block IDs to blocks
        block_map = {block['Id']: block for block in blocks}
        
        for block in blocks:
            if block.get('BlockType') == 'KEY_VALUE_SET':
                entity_type = block.get('EntityTypes', [])
                
                if 'KEY' in entity_type:
                    key_text = self.get_text_from_relationships(block, block_map, 'CHILD')
                    
                    # Find the corresponding VALUE
                    value_blocks = self.get_related_blocks(block, block_map, 'VALUE')
                    for value_block in value_blocks:
                        value_text = self.get_text_from_relationships(value_block, block_map, 'CHILD')
                        if key_text and value_text:
                            key_value_pairs[key_text.lower().strip()] = value_text.strip()
        
        return key_value_pairs
    
    def extract_tables(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract table data from Textract"""
        tables = []
        block_map = {block['Id']: block for block in blocks}
        
        for block in blocks:
            if block.get('BlockType') == 'TABLE':
                table_data = self.parse_table_block(block, block_map)
                if table_data:
                    tables.append(table_data)
        
        return tables
    
    def parse_table_block(self, table_block: Dict[str, Any], block_map: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single table block"""
        try:
            rows = {}
            
            # Get all cell blocks for this table
            cell_blocks = self.get_related_blocks(table_block, block_map, 'CHILD')
            
            for cell_block in cell_blocks:
                if cell_block.get('BlockType') == 'CELL':
                    row_index = cell_block.get('RowIndex', 0)
                    col_index = cell_block.get('ColumnIndex', 0)
                    
                    if row_index not in rows:
                        rows[row_index] = {}
                    
                    cell_text = self.get_text_from_relationships(cell_block, block_map, 'CHILD')
                    rows[row_index][col_index] = cell_text or ''
            
            # Convert to list format
            table_rows = []
            for row_index in sorted(rows.keys()):
                row_data = []
                for col_index in sorted(rows[row_index].keys()):
                    row_data.append(rows[row_index][col_index])
                table_rows.append(row_data)
            
            return {
                'rows': table_rows,
                'row_count': len(table_rows),
                'column_count': max(len(row) for row in table_rows) if table_rows else 0
            }
            
        except Exception as e:
            logger.error(f"Error parsing table: {str(e)}")
            return None
    
    def get_related_blocks(self, block: Dict[str, Any], block_map: Dict[str, Any], relationship_type: str) -> List[Dict[str, Any]]:
        """Get blocks related through relationships"""
        related_blocks = []
        
        relationships = block.get('Relationships', [])
        for relationship in relationships:
            if relationship.get('Type') == relationship_type:
                for block_id in relationship.get('Ids', []):
                    if block_id in block_map:
                        related_blocks.append(block_map[block_id])
        
        return related_blocks
    
    def get_text_from_relationships(self, block: Dict[str, Any], block_map: Dict[str, Any], relationship_type: str) -> str:
        """Get text from related blocks"""
        text_parts = []
        
        related_blocks = self.get_related_blocks(block, block_map, relationship_type)
        for related_block in related_blocks:
            if related_block.get('BlockType') == 'WORD':
                text = related_block.get('Text', '')
                if text:
                    text_parts.append(text)
        
        return ' '.join(text_parts)
    
    def extract_invoice_fields_with_patterns(self, text: str) -> Dict[str, str]:
        """Extract invoice fields using regex patterns"""
        fields = {}
        text_lower = text.lower()
        
        for field_name, patterns in self.invoice_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                if match:
                    fields[field_name] = match.group(1).strip()
                    break  # Use first match
        
        return fields
    
    def merge_extracted_fields(self, pattern_fields: Dict[str, str], kv_fields: Dict[str, str]) -> Dict[str, str]:
        """Merge fields from pattern matching and key-value extraction"""
        merged = pattern_fields.copy()
        
        # Key-value pairs take precedence if they exist
        field_mappings = {
            'invoice number': 'invoice_id',
            'invoice #': 'invoice_id',
            'invoice id': 'invoice_id',
            'total': 'amount',
            'amount due': 'amount',
            'due date': 'due_date',
            'payment due': 'due_date',
            'vendor': 'vendor',
            'from': 'vendor',
            'bill from': 'vendor'
        }
        
        for kv_key, kv_value in kv_fields.items():
            normalized_key = kv_key.lower().strip()
            if normalized_key in field_mappings:
                field_name = field_mappings[normalized_key]
                merged[field_name] = kv_value
        
        return merged
    
    def parse_amount(self, amount_str: Optional[str]) -> Optional[float]:
        """Parse amount string to float"""
        if not amount_str:
            return None
        
        try:
            # Remove currency symbols and spaces
            cleaned = re.sub(r'[\$\s,]', '', amount_str.strip())
            # Replace comma with dot for decimal separator
            cleaned = cleaned.replace(',', '.')
            return float(cleaned)
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse amount: {amount_str}")
            return None
    
    def parse_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse date string to ISO format"""
        if not date_str:
            return None
        
        try:
            from dateutil import parser
            parsed_date = parser.parse(date_str, fuzzy=True)
            return parsed_date.strftime('%Y-%m-%d')
        except Exception:
            logger.warning(f"Could not parse date: {date_str}")
            return None
    
    def calculate_field_confidence(self, blocks: List[Dict[str, Any]], fields: Dict[str, str]) -> Dict[str, float]:
        """Calculate confidence scores for extracted fields"""
        confidence_scores = {}
        
        # Calculate average confidence from all word blocks
        word_confidences = []
        for block in blocks:
            if block.get('BlockType') == 'WORD' and 'Confidence' in block:
                word_confidences.append(block['Confidence'])
        
        avg_confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 0
        
        # Assign confidence scores to fields
        for field_name in fields:
            confidence_scores[field_name] = avg_confidence
        
        return confidence_scores
    
    def calculate_average_confidence(self, textract_result: Dict[str, Any]) -> float:
        """Calculate average confidence score"""
        blocks = textract_result.get('Blocks', [])
        confidences = []
        
        for block in blocks:
            if 'Confidence' in block:
                confidences.append(block['Confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def store_structured_data(self, bucket: str, key: str, structured_data: Dict[str, Any]) -> None:
        """Store structured data to S3"""
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(structured_data, default=str),
                ContentType='application/json'
            )
            logger.info(f"Stored structured data: s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"Error storing structured data: {str(e)}")
            # Don't raise - this is not critical

# Initialize the processor
textract_processor = PayScanTextractProcessor()

def lambda_handler(event, context):
    """Lambda entry point"""
    return textract_processor.lambda_handler(event, context)_textract_output(self, bucket: str, key: str, textract_result: Dict[str, Any]) -> None:
        """Store raw Textract output to S3"""
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(textract_result, default=str),
                ContentType='application/json'
            )
            logger.info(f"Stored Textract output: s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"Error storing Textract output: {str(e)}")
            # Don't raise - this is not critical
    
    def store