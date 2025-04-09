#!/usr/bin/env python3
import xmlrpc.client

url = 'https://www.pricepaper.com'
db = 'ppt-apps15'
username = ''
password = ''


# Define the folder id you want to create the document
folder_id = 7 # Assign the corresponding folder ID, for finance folder ID=2

if not url or not db or not password:
    print('URL or DB or Password missing.')

# Connect to the Odoo server
common = xmlrpc.client.ServerProxy(f'{url}/xmlrpc/2/common')
uid = common.authenticate(db, username, password, {})
models = xmlrpc.client.ServerProxy(f'{url}/xmlrpc/2/object')




# Search for attachments in invoices starting with 'delivery%'
attachments = models.execute_kw(db, uid, password, 'ir.attachment', 'search_read', [
    [('res_model', '=', 'account.move'), ('name', '=like', 'INV-%')]
], {
    'fields': ['id']
})
# Check if any attachments were found
if not attachments:
    print("No attachments found with names starting with 'INV%' in invoices.")
else:
    # Loop through each attachment found
    for attachment in attachments:
        try:
            # Define values for creating a new document linked to the attachment
            document_vals = {
                'attachment_id': attachment['id'],
                'folder_id': folder_id,
                'active': True,
            }
            # Create the document in the 'documents.document' model
            document_id = models.execute_kw(db, uid, password, 'documents.document', 'create', [document_vals])
            print(f"Created document for attachment ID: {attachment['id']}")
        except xmlrpc.client.Fault as e:
            # Handle cases where the document already exists
            if "This attachment is already a document" in str(e):
                print(f"Attachment {attachment['id']} already has a document. Skipping.")
            elif 'documents_document_attachment_unique' in str(e):
                print(f"Attachment {attachment['id']} already has a document. Skipping.")
            else:
                # Print any other errors encountered
                print(f"Error: {str(e)}")
