# Google Drive Checkpoints Storage

Google Drive can be used as an external storage for checkpoints and configs.

A Google Drive external storage config supports the following parameters:
- `storage_dir_id`: \
The ID of the Google Drive folder where experiments are stored. It can be obtained from a URL:
`https://drive.google.com/drive/u/1/folders/<folder_id>`
- `client_credentials_filepath`: \
Path to the file with OAuth client credentials (id and secret). These credentials are used to obtain an access token through user authentication. By default, "client_secrets.json".
- `service_account_key_filepath`: \
Path to the JSON file with service account credentials. These credentials are used to obtain an access token. Supports passing a URL. By default, a link to my service account.

This config is specified either as a part of a training config:
```
"trainer": {
  "external_storage": {
    "type": "GDriveStorage",
    "args": {
      "storage_folder_id": "<directory id>",
    }
  }
}
```
or as a separate config for using the [script for checkpoints download](../scripts/model_loader) ([example](../gdrive_storage/external_storage.json)).

## Access to shared checkpoints

If you only want to download publicly available (shared via link) checkpoints, a *service account* will suffice.
You can use a service account that I created ([link to credentials](https://drive.google.com/uc?export=download&id=1stvKJB9Kuoh9vbpeBlGfBI5foHIGtmME)).
By default, it will be used automatically in the code.
Also, you can use a service account to export checkpoints to its Google Drive (not your personal one) (**TODO:** check and add instructions).

> You can read more about service accounts briefly [here](https://developers.google.com/identity/protocols/oauth2#serviceaccount) and in more detail [here](https://cloud.google.com/iam/docs/service-account-overview).

## Access to a personal Google Drive

If you want to export checkpoints to your personal Google Drive or download not shared checkpoints from it, you will need to grant access to it:
1. Create an application using Google Drive API with its client credentials via the Google Cloud console (you can consider the code from this repo an application that will use Google Drive API). To do so, follow the [instruction](https://pythonhosted.org/PyDrive/quickstart.html#authentication) and download a JSON file with credentials.
2. Specify the path to the file with client credentials as `client_credentials_filepath` in the external storage config.
3. If you run the code using Google Drive API, you will be asked to authenticate to a Google account and grant full access to its Google Drive. Follow the instructions in the console. As a result, an access token will be received from the Google server. With this token, your program will be able to import and export checkpoints using your Google Drive. Note that neither credentials nor access tokens are sent anywhere except your host and Google servers, as you may check in the code.

> You can read more about credentials, secrets, and access token in the [official overview](https://developers.google.com/identity/protocols/oauth2).