export const ACCEPTED = ['.dcm', '.nii', '.nii.gz', '.zip', '.png', '.jpg', '.jpeg'];
export const MAX_BYTES = 500 * 1024 * 1024;

export function formatBytes(n) {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), units.length - 1);
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function validateFile(file) {
  if (!file) return 'Choose a CT scan file to continue.';
  const name = file.name.toLowerCase();
  if (!ACCEPTED.some((ext) => name.endsWith(ext)))
    return `Unsupported file type. Use ${ACCEPTED.join(', ')}.`;
  if (file.size > MAX_BYTES) return `File is too large (${formatBytes(file.size)}). The limit is 500 MB.`;
  if (file.size === 0) return 'This file is empty. Choose another file.';
  return null;
}
