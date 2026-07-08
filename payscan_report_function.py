"""
PayScan AI Agent - Report Generation Function
Generates processing summary reports
"""

import json
import boto3
import os
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

class PayScanReportGenerator:
    def __init__(self):
        self.results_bucket = os.environ.get('RESULTS_BUCKET', 'payscan-results-prod')
    
    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Main Lambda handler for report generation"""
        try:
            logger.info(f"Starting report generation: {json.dumps(event, default=str)}")
            
            job_id = event['job_id']
            processing_results = event['processing_results']
            
            # Generate comprehensive report
            report = self.build_report(job_id, processing_results)
            
            # Store report
            report_key = f"reports/{job_id}/processing_report.json"
            self.store_report(report_key, report)
            
            # Generate text summary
            summary_key = f"reports/{job_id}/summary.txt"
            summary_text = self.generate_summary_text(report)
            self.store_text_report(summary_key, summary_text)
            
            return {
                'success': True,
                'job_id': job_id,
                'report_location': f"s3://{self.results_bucket}/{report_key}",
                'summary_location': f"s3://{self.results_bucket}/{summary_key}",
                'summary': report.get('invoice_summary', {}),
                'alerts_count': len(report.get('alerts', [])),
                'recommendations_count': len(report.get('recommendations', [])),
                'generation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {
                'success': False,
                'job_id': event.get('job_id'),
                'error': str(e),
                'error_type': 'report_generation_error'
            }
    
    def build_report(self, job_id: str, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive processing report"""
        try:
            report = {
                'job_id': job_id,
                'report_timestamp': datetime.utcnow().isoformat(),
                'processing_summary': {
                    'textract_success': processing_results.get('textract', {}).get('success', False),
                    'sagemaker_success': processing_results.get('sagemaker', {}).get('success', False),
                    'bedrock_success': processing_results.get('bedrock', {}).get('success', False),
                    'storage_success': processing_results.get('storage', {}).get('success', False),
                    'duplicate_check_success': processing_results.get('duplicate_check', {}).get('success', False)
                },
                'invoice_summary': {},
                'textract_summary': {},
                'alerts': [],
                'duplicate_alerts': [],
                'recommendations': [],
                'quality_metrics': {}
            }
            
            # Extract invoice summary from Bedrock
            bedrock_payload = processing_results.get('bedrock', {}).get('Payload', {})
            if bedrock_payload.get('success'):
                curated_data = bedrock_payload.get('curated_invoice_data', {})
                report['invoice_summary'] = {
                    'invoice_id': curated_data.get('invoice_id'),
                    'vendor_name': curated_data.get('vendor_name'),
                    'amount': curated_data.get('amount'),
                    'due_date': curated_data.get('due_date'),
                    'invoice_date': curated_data.get('invoice_date'),
                    'status': curated_data.get('status'),
                    'risk_level': curated_data.get('overall_risk_level'),
                    'data_quality_score': curated_data.get('data_quality_score')
                }
                
                # Extract alerts
                report['alerts'] = curated_data.get('alerts', [])
                report['recommendations'] = curated_data.get('recommendations', [])
            
            # Extract Textract summary
            textract_payload = processing_results.get('textract', {}).get('Payload', {})
            if textract_payload.get('success'):
                report['textract_summary'] = {
                    'total_blocks': textract_payload.get('processing_stats', {}).get('total_blocks', 0),
                    'confidence_score': textract_payload.get('processing_stats', {}).get('confidence_score', 0),
                    'pages_processed': textract_payload.get('processing_stats', {}).get('pages_processed', 0)
                }
            
            # Extract duplicate alerts
            duplicate_payload = processing_results.get('duplicate_check', {}).get('Payload', {})
            if duplicate_payload.get('success') and duplicate_payload.get('has_duplicates'):
                for duplicate in duplicate_payload.get('duplicates_found', []):
                    report['duplicate_alerts'].append({
                        'type': 'duplicate',
                        'severity': 'high',
                        'title': 'Potential Duplicate Invoice',
                        'existing_invoice_id': duplicate.get('existing_invoice_id'),
                        'existing_job_id': duplicate.get('existing_job_id'),
                        'confidence': duplicate.get('confidence'),
                        'match_reasons': duplicate.get('match_reasons', [])
                    })
            
            # Calculate quality metrics
            report['quality_metrics'] = self.calculate_metrics(processing_results)
            
            return report
            
        except Exception as e:
            logger.error(f"Error building report: {str(e)}")
            return {'error': str(e)}
    
    def calculate_metrics(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate processing quality metrics"""
        try:
            successful_stages = sum(1 for stage in processing_results.values() 
                                   if isinstance(stage, dict) and stage.get('Payload', {}).get('success', False))
            total_stages = len([s for s in processing_results.values() if isinstance(s, dict) and 'Payload' in s])
            
            success_rate = (successful_stages / total_stages * 100) if total_stages > 0 else 0
            
            # Get data quality score
            bedrock_payload = processing_results.get('bedrock', {}).get('Payload', {})
            data_quality = bedrock_payload.get('curated_invoice_data', {}).get('data_quality_score', 0)
            
            return {
                'success_rate': f"{success_rate:.1f}%",
                'successful_stages': successful_stages,
                'total_stages': total_stages,
                'data_quality_score': data_quality,
                'processing_complete': successful_stages == total_stages
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def generate_summary_text(self, report: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        try:
            lines = [
                "=" * 60,
                "PAYSCAN INVOICE PROCESSING REPORT",
                "=" * 60,
                "",
                f"Job ID: {report.get('job_id')}",
                f"Generated: {report.get('report_timestamp')}",
                "",
                "INVOICE SUMMARY",
                "-" * 60,
            ]
            
            summary = report.get('invoice_summary', {})
            lines.extend([
                f"Invoice ID: {summary.get('invoice_id', 'N/A')}",
                f"Vendor: {summary.get('vendor_name', 'N/A')}",
                f"Amount: ${summary.get('amount', 'N/A')}",
                f"Due Date: {summary.get('due_date', 'N/A')}",
                f"Status: {summary.get('status', 'N/A')}",
                f"Risk Level: {summary.get('risk_level', 'N/A')}",
                f"Data Quality Score: {summary.get('data_quality_score', 'N/A')}",
                "",
                "PROCESSING METRICS",
                "-" * 60,
            ])
            
            metrics = report.get('quality_metrics', {})
            lines.extend([
                f"Success Rate: {metrics.get('success_rate', 'N/A')}",
                f"Stages Completed: {metrics.get('successful_stages', 0)}/{metrics.get('total_stages', 0)}",
                "",
                "ALERTS",
                "-" * 60,
            ])
            
            alerts = report.get('alerts', [])
            if alerts:
                for alert in alerts[:5]:  # Show first 5
                    lines.append(f"• [{alert.get('severity', 'INFO').upper()}] {alert.get('title', 'Alert')}")
            else:
                lines.append("No alerts")
            
            lines.extend([
                "",
                "DUPLICATE CHECKS",
                "-" * 60,
            ])
            
            duplicates = report.get('duplicate_alerts', [])
            if duplicates:
                for dup in duplicates:
                    lines.append(f"• Found: {dup.get('existing_invoice_id')} (Confidence: {dup.get('confidence')})")
            else:
                lines.append("No duplicates detected")
            
            lines.append("=" * 60)
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error generating text summary: {str(e)}")
            return "Error generating summary"
    
    def store_report(self, key: str, report: Dict[str, Any]) -> None:
        """Store JSON report"""
        try:
            s3_client.put_object(
                Bucket=self.results_bucket,
                Key=key,
                Body=json.dumps(report, default=str, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Stored report: s3://{self.results_bucket}/{key}")
        except Exception as e:
            logger.error(f"Error storing report: {str(e)}")
            raise
    
    def store_text_report(self, key: str, text: str) -> None:
        """Store text summary"""
        try:
            s3_client.put_object(
                Bucket=self.results_bucket,
                Key=key,
                Body=text,
                ContentType='text/plain'
            )
            logger.info(f"Stored text summary: s3://{self.results_bucket}/{key}")
        except Exception as e:
            logger.error(f"Error storing text summary: {str(e)}")
            raise

generator = PayScanReportGenerator()

def lambda_handler(event, context):
    """Lambda entry point"""
    return generator.lambda_handler(event, context)
