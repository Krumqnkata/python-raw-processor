"""Read selected EXIF fields and embed safe metadata without rewriting pixels."""
from pathlib import Path
import struct
import zlib
import exifread
from PIL import Image, TiffImagePlugin

FIELDS = {'Image Make':271,'Image Model':272,'EXIF LensModel':42036,
          'Image DateTime':306,'EXIF DateTimeOriginal':36867,'EXIF DateTimeDigitized':36868,
          'EXIF ExposureTime':33434,'EXIF FNumber':33437,'EXIF ISOSpeedRatings':34855,
          'EXIF FocalLength':37386,'EXIF ExposureBiasValue':37380}


def read_metadata(path):
    with Path(path).open('rb') as handle:
        tags = exifread.process_file(handle,details=False,strict=False)
    result = {}
    for key in FIELDS:
        if key not in tags:
            continue
        tag = tags[key]
        if tag.field_type in {2,7}:
            value = str(tag).strip('\x00')[:512]
        else:
            vals = tag.values
            value = vals[0] if isinstance(vals,list) and vals else vals
            if hasattr(value,'num'):
                value = [int(value.num),int(value.den)]
            if not isinstance(value,(str,int,float,list)):
                value = str(value)
        result[key] = value
    return result


def exif_bytes(data, params, size):
    exif = Image.Exif()
    sub = {40961:1,40962:size[0],40963:size[1]}
    exif[274] = 1  # Pixels already reflect RAW orientation, rotation and crop.
    exif[305] = 'RAW Studio'
    for key,value in data.items():
        if key not in FIELDS:
            continue
        is_camera = key in {'Image Make','Image Model','EXIF LensModel'}
        is_date = 'DateTime' in key
        if (is_camera and not params.exif_camera) or (is_date and not params.exif_date) or (not is_camera and not is_date and not params.exif_exposure):
            continue
        if isinstance(value,list) and len(value)==2:
            if value[1]==0: continue
            value = TiffImagePlugin.IFDRational(*value)
        (sub if key.startswith('EXIF ') else exif)[FIELDS[key]] = value
    exif[34665] = sub
    return exif.tobytes()


def embed_exif(encoded, extension, exif):
    """Insert JPEG APP1 or PNG eXIf chunks, preserving PNG16 sample precision."""
    data = encoded.tobytes() if hasattr(encoded,'tobytes') else bytes(encoded)
    if extension == '.jpg':
        if len(exif)+2 > 65535:
            raise ValueError('EXIF блокът е твърде голям.')
        return data[:2]+b'\xff\xe1'+struct.pack('>H',len(exif)+2)+exif+data[2:]
    payload = exif[6:] if exif.startswith(b'Exif\x00\x00') else exif
    chunk = b'eXIf'+payload
    block = struct.pack('>I',len(payload))+chunk+struct.pack('>I',zlib.crc32(chunk)&0xffffffff)
    # Insert after IHDR, before image data.
    offset = 8+12+struct.unpack('>I',data[8:12])[0]
    return data[:offset]+block+data[offset:]
