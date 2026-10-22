import { describe, it, expect } from 'vitest';
import { formatBytes, validateFile } from './files.js';

describe('formatBytes', () => {
  it('formats sizes', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1536)).toBe('1.5 KB');
  });
});

describe('validateFile', () => {
  it('accepts supported files', () => {
    expect(validateFile({ name: 'scan.nii.gz', size: 1000 })).toBeNull();
    expect(validateFile({ name: 'slice.DCM', size: 1000 })).toBeNull();
  });
  it('rejects bad files', () => {
    expect(validateFile(null)).toMatch(/Choose/);
    expect(validateFile({ name: 'a.exe', size: 10 })).toMatch(/Unsupported/);
    expect(validateFile({ name: 'a.dcm', size: 600 * 1024 * 1024 })).toMatch(/too large/);
    expect(validateFile({ name: 'a.dcm', size: 0 })).toMatch(/empty/);
  });
});
