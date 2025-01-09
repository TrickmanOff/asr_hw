"""
Class for importing and exporting models from Google Drive.
"""
import dataclasses
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import urllib
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

from oauth2client.service_account import ServiceAccountCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from hw_asr.storage.experiments_storage import RunStorage
from hw_asr.storage.external_storage import CheckpointInfo, ExternalStorage, RunInfo
from hw_asr.utils.util import ROOT_PATH, download_file


logger = logging.getLogger(__name__)

ARCHIVE_FORMAT = 'zip'
SERVICE_ACCOUNT_CREDENTIALS_URL = 'https://drive.google.com/uc?export=download&id=1stvKJB9Kuoh9vbpeBlGfBI5foHIGtmME'


def _archive_file(archive_filepath: str, filepath: str) -> str:
    """
    returns archive filepath
    """
    if os.path.isdir(filepath):
        shutil.make_archive(archive_filepath, ARCHIVE_FORMAT, root_dir=filepath)
    else:
        parent_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        shutil.make_archive(archive_filepath, ARCHIVE_FORMAT, root_dir=parent_dir, base_dir=filename)
    return archive_filepath + '.' + ARCHIVE_FORMAT


# It is a little more secure than storing credentials directly in the repository.
# Also, most importantly, it helps to avoid service account freeze by Google due to automatic credentials leakage checks.
def get_or_load_service_account_credentials(url: str) -> str:
    """
    :return: filepath with downloaded service account credentials
    """
    # getting URL hash
    h = hashlib.sha1()
    h.update(url.encode('utf-8'))
    url_hash = h.hexdigest()[:10]
    credentials_filepath = f'service_account_credentials_{url_hash}.json'
    if os.path.exists(credentials_filepath):
        print(f'Already downloaded service account credentials "{credentials_filepath}" will be used')
    else:
        print(f'Downloading service account credentials: "{url}" -> "{credentials_filepath}"')
        download_file(url, to_filename=credentials_filepath)
    return credentials_filepath


def extract_auth_code_from_url(url: str) -> str:
    pattern = '\?code=(.*)\&'
    match = re.search(pattern, url)
    auth_code = match.groups()[0]
    return urllib.parse.unquote(auth_code)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


class GDriveStorage(ExternalStorage):
    CONFIG_FILENAME = 'config.json'
    CHECKPOINT_EXT = '.pth'

    def __init__(self, storage_folder_id: str,
                 client_credentials_filepath: str = './client_secrets.json',
                 service_account_key_filepath: Optional[str] = SERVICE_ACCOUNT_CREDENTIALS_URL,
                 save_credentials: bool = True,
                 gauth=None):
        """
        :param storage_folder_id: the id of the Google Drive folder where experiments are stored.
            It can be obtained from a URL: https://drive.google.com/drive/u/1/folders/<folder_id>
        :param client_credentials_filepath: path to the file with OAuth client credentials (id and secret).
            These credentials are used to obtain an access token through user authentication.
        :param service_account_key_filepath: path to the JSON file with service account credentials.
            These credentials are used to obtain an access token. Passing a URL is supported.
        """
        super().__init__()
        if service_account_key_filepath is not None:
            assert gauth is None
            if os.path.exists(service_account_key_filepath):
                pass
            elif os.path.exists(ROOT_PATH / service_account_key_filepath):
                service_account_key_filepath = ROOT_PATH / service_account_key_filepath
            else:
                service_account_key_filepath = get_or_load_service_account_credentials(service_account_key_filepath)

            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                service_account_key_filepath,
                scopes = ['https://www.googleapis.com/auth/drive.readonly'],
            )
            gauth = GoogleAuth()
            gauth.credentials = credentials

        if gauth is None:
            # TODO: think of a cleaner solution
            # this code leads to saving and loading an access token after authenticating once in order to avoid manual authentication each time
            GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = client_credentials_filepath
            GoogleAuth.DEFAULT_SETTINGS['save_credentials'] = save_credentials
            GoogleAuth.DEFAULT_SETTINGS['save_credentials_backend'] = 'file'
            save_credentials_file = Path(client_credentials_filepath)
            save_credentials_file = save_credentials_file.with_stem(save_credentials_file.stem + '_auth')
            GoogleAuth.DEFAULT_SETTINGS['save_credentials_file'] = save_credentials_file

            gauth = GoogleAuth()
            if save_credentials and save_credentials_file.exists():
                gauth.LoadCredentials()
            # Create local webserver and handle authentication automatically
            # gauth.LocalWebserverAuth()
            if gauth.access_token_expired:
                msg = 'No prior authentication.' if gauth.credentials is None else 'Access token expired.'
                msg += ' Authentication required.'
                print(msg)

                url = gauth.GetAuthUrl()
                print(f'Visit the url:\n{url}')
                redirected_url = input('Enter the url you were redirected to after authentication:\n')
                auth_code = extract_auth_code_from_url(redirected_url)
                gauth.Auth(auth_code)
                if save_credentials:
                    gauth.SaveCredentialsFile(save_credentials_file)
            # gauth.CommandLineAuth()
        self.drive = GoogleDrive(gauth)
        self.storage_folder_id = storage_folder_id

    # def _get_part_name(self, part: ModelParts):
    #     if part not in self.PARTS_NAMES:
    #         raise RuntimeError(f'Model part {part} is not supported by GDriveStorage')
    #     return self.PARTS_NAMES[part]

    def _import_config(self, run_storage: RunStorage) -> None:
        self._download_file(run_storage, self.CONFIG_FILENAME, run_storage.get_config_filepath())
        logger.info(f'Config for run {run_storage.get_run_id()} imported')

    def _import_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        checkpoint_filename = checkpoint_name + self.CHECKPOINT_EXT
        to_filepath = run_storage.get_checkpoints_dirpath() / checkpoint_filename
        self._download_file(run_storage, checkpoint_filename, to_filepath)
        logger.info(f'Checkpoint {checkpoint_name} for run {run_storage.exp_name}:{run_storage.run_name} imported')

    def _export_config(self, run_storage: RunStorage) -> None:
        config_local_filepath = run_storage.get_config_filepath()
        config_local_filename = os.path.basename(config_local_filepath)
        run_drive_dir = self._get_run_dir(run_storage.exp_name, run_storage.run_name)
        self._upload_file(run_drive_dir, config_local_filename, config_local_filepath)
        logger.info(f'Config for run {run_storage.exp_name}:{run_storage.run_name} exported')

    def _export_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        checkpoint_filename = checkpoint_name + self.CHECKPOINT_EXT
        checkpoint_filepath = run_storage.get_checkpoints_dirpath() / checkpoint_filename
        run_drive_dir = self._get_run_dir(run_storage.exp_name, run_storage.run_name)
        self._upload_file(run_drive_dir, checkpoint_filename, checkpoint_filepath)

    def _get_subdir(self, parent_dir_id: str, subdir_name: str) -> str:
        """
        :return: id
        """
        query = f'"{parent_dir_id}" in parents and title="{subdir_name}" and trashed=false'
        files = self.drive.ListFile({'q': query}).GetList()
        if len(files) != 0:
            return files[0]['id']
        else:
            file_metadata = {
                'title': subdir_name,
                'parents': [{'id': parent_dir_id}],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.drive.CreateFile(file_metadata)
            folder.Upload()
            return folder['id']

    def _get_experiment_dir(self, exp_name: str) -> str:
        """
        Creates if it does not exist
        :return: id
        """
        return self._get_subdir(parent_dir_id=self.storage_folder_id, subdir_name=exp_name)

    def _get_run_dir(self, exp_name: str, run_name: str) -> str:
        """
        Creates if it does not exist
        :return: id
        """
        exp_dir = self._get_experiment_dir(exp_name)
        return self._get_subdir(parent_dir_id=exp_dir, subdir_name=run_name)

    # def _import_model_part(self, model_name: str, part: ModelParts, to_filepath: str) -> None:
    #     part_name = self._get_part_name(part)
    #     model_dir = self._get_model_dir(model_name)
    #     with tempfile.TemporaryDirectory() as archive_dir:
    #         downloaded_archive_filepath = os.path.join(archive_dir,
    #                                                    model_name + '.' + ARCHIVE_FORMAT)
    #         query = f'"{model_dir}" in parents and title="{part_name}" and trashed=false'
    #         files = self.drive.ListFile({'q': query}).GetList()
    #         if len(files) == 0:
    #             raise RuntimeError(f'No part {part} found')
    #         archive_file = files[0]
    #         archive_file.GetContentFile(downloaded_archive_filepath)
    #
    #         if os.path.exists(to_filepath) and os.path.isdir(to_filepath):
    #             shutil.unpack_archive(downloaded_archive_filepath, to_filepath, ARCHIVE_FORMAT)
    #         else:
    #             to_dirpath = os.path.dirname(to_filepath)
    #             imported_filename = os.path.basename(to_filepath)
    #             shutil.unpack_archive(downloaded_archive_filepath, to_dirpath, ARCHIVE_FORMAT)
    #             unpacked_filepath = os.path.join(to_dirpath, os.path.basename(downloaded_archive_filepath))
    #             os.rename(unpacked_filepath, os.path.join(to_dirpath, imported_filename))

    def _upload_file(self, to_dir_id: str, to_filename: str, filepath: Union[str, Path]) -> None:
        query = f"'{to_dir_id}' in parents and title='{to_filename}' and trashed=false"
        files = self.drive.ListFile({'q': query}).GetList()
        if len(files) != 0:
            for file in files:
                file.Delete()

        file = self.drive.CreateFile(
            {'title': to_filename, 'parents': [{'id': to_dir_id}]})
        file.SetContentFile(filepath)
        file.Upload()

    def _download_file(self, run_storage: RunStorage, drive_filename: str, to_filepath: Union[str, Path]):
        run_drive_dir = self._get_run_dir(run_storage.exp_name, run_storage.run_name)
        query = f'"{run_drive_dir}" in parents and title="{drive_filename}" and trashed=false'
        files = self.drive.ListFile({'q': query}).GetList()
        if len(files) == 0:
            raise RuntimeError(f'No file {drive_filename} for run {run_storage.get_run_id()}')
        file = files[0]

        if os.path.exists(to_filepath):
            logger.warning(f'File {os.path.basename(to_filepath)} for run {run_storage.get_run_id()} will be overriden')

        file.GetContentFile(to_filepath)
        logger.info(f'Successfully downloaded file {drive_filename} for run {run_storage.get_run_id()}')

    def _export_dir(self, to_dir_id: str, from_dirpath: str) -> None:
        """
        Without archiving, `from_dirpath` must not contain directories
        """
        for filename in os.listdir(from_dirpath):
            filepath = os.path.join(from_dirpath, filename)
            self._upload_file(to_dir_id, filename, filepath)

    def _export_as_archive(self, dir_id: str, archive_name: str, from_path: str) -> None:
        with tempfile.TemporaryDirectory() as archive_dir:
            archive_filepath = _archive_file(os.path.join(archive_dir, archive_name),
                                             filepath=from_path)
            self._upload_file(dir_id, archive_name, archive_filepath)

    # def _export_model_part(self, model_name: str, part: ModelParts, from_path: str) -> None:
    #     part_name = self._get_part_name(part)
    #     model_dir = self._get_model_dir(model_name)
    #
    #     # extra copy for convenience
    #     if part is ModelParts.CONFIG:
    #         config_subdir = self._get_subdir(parent_dir_id=model_dir, subdir_name=self.CONFIG_SUBDIR_NAME)
    #         self._export_dir(config_subdir, from_path)
    #
    #     self._upload_file(model_dir, part_name, from_path)
    #     # self._export_as_archive(model_dir, part_name, from_path)
    #

    def get_available_runs(self) -> Dict[str, Dict[str, RunInfo]]:
        exps_list = self.drive.ListFile(
            {'q': f"'{self.storage_folder_id}' in parents and trashed=false"}).GetList()

        res = {}

        for exp in exps_list:
            runs_list = self.drive.ListFile(
                {'q': f"'{exp['id']}' in parents and trashed=false"}).GetList()
            exp_runs = {}
            for run in runs_list:
                run_info = RunInfo([], False)
                run_files_list = self.drive.ListFile(
                    {'q': f"'{run['id']}' in parents and trashed=false"}).GetList()
                for run_file in run_files_list:
                    run_filename = run_file['title']
                    if run_filename.endswith(self.CHECKPOINT_EXT):
                        checkpoint_name = os.path.splitext(run_filename)[0]
                        checkpoint = CheckpointInfo(name=checkpoint_name, creation_date=run_file['createdDate'])
                        run_info.checkpoints.append(checkpoint)
                    elif run_filename == self.CONFIG_FILENAME:
                        run_info.with_config = True
                exp_runs[run['title']] = run_info
            res[exp['title']] = exp_runs

        return res

    def list_content(self) -> str:
        runs = self.get_available_runs()
        return json.dumps(runs, indent=4, cls=EnhancedJSONEncoder)

    def _import_model(self, model_name: str, to_dirpath: str) -> None:
        with tempfile.TemporaryDirectory() as archive_dir:
            downloaded_archive_filepath = os.path.join(archive_dir,
                                                       model_name + '.' + ARCHIVE_FORMAT)
            query = f'"{self.storage_folder_id}" in parents and title="{model_name}.{ARCHIVE_FORMAT}" and trashed=false'
            archive_file = self.drive.ListFile({'q': query}).GetList()[0]
            archive_file.GetContentFile(downloaded_archive_filepath)

            shutil.unpack_archive(downloaded_archive_filepath, to_dirpath, ARCHIVE_FORMAT)
