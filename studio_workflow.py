"""Simple workflow policies and immutable snapshots for background recovery saves."""
from dataclasses import asdict, replace
from pathlib import Path
import os
from datetime import datetime
from studio import EDIT_FIELDS, EXPORT_FIELDS, atomic_json, read_json
from raw_engine import ProcessingParams

EXPORT_PRESETS = {
    'За споделяне': dict(output_format='jpg', quality=90, max_edge=2048),
    'Пълна резолюция': dict(output_format='jpg', quality=95, max_edge=0),
    'За последваща обработка': dict(output_format='png', png_bit_depth=16, max_edge=0),
}
COPY_GROUPS = {
    'Всички общи корекции': EDIT_FIELDS,
    'Само светлина': {'auto_exposure','exposure_ev','tone_mapping','tone_strength','clahe_clip','shadows','highlights','whites','blacks'},
    'Само цвят': {'white_balance','wb_gains','temperature','tint','saturation','vibrance','monochrome'},
    'Само детайл': {'denoise','denoise_strength','sharpen','sharpen_amount'},
}


def automatic_params(current):
    """Natural automatic corrections; keep export policy and individual geometry."""
    defaults = ProcessingParams()
    return replace(current, **{k:getattr(defaults,k) for k in EDIT_FIELDS})


def config_directory():
    return Path(os.environ.get('RAW_STUDIO_CONFIG_DIR', str(Path.home()/'.raw-studio'))).expanduser().resolve()


def load_preferences(folder):
    try:
        data = read_json(Path(folder)/'preferences.json')
        return data if isinstance(data,dict) and data.get('version')==1 else {}
    except (OSError,ValueError,TypeError):
        return {}


def session_snapshot(session, target):
    """Read state on the Tk thread; immutable params/history are safe for a worker."""
    return (Path(target).resolve(), session.defaults, str(session.output), str(session.selected),
            str(session.project_path or ''),
            tuple((p.path,p.params,p.included,p.rating,tuple(p.history),p.cursor) for p in session.photos.values()))


def write_snapshot(snapshot):
    path,defaults,output,selected,origin,photos = snapshot
    def relative(value):
        if not value: return ''
        try: return os.path.relpath(value,path.parent)
        except ValueError: return str(value)
    atomic_json(path, {'version':1,'defaults':asdict(defaults),'output':output,
                      'selected':relative(selected),'project_origin':origin,
                      'photos':[{'path':relative(source),'params':asdict(params),'included':included,
                                 'rating':rating,'history':[asdict(p) for p in history],'cursor':cursor}
                                for source,params,included,rating,history,cursor in photos]})


def export_description(params, count, source=None):
    size = f'{params.max_edge} px' if params.max_edge else 'пълна резолюция'
    quality = f'качество {params.quality}%' if params.output_format=='jpg' else f'{params.png_bit_depth} бита'
    example = params.filename_template.format(stem=Path(source).stem if source else 'снимка',index=1,date=datetime.now().strftime('%Y%m%d'),camera='camera')
    return f'{count} снимки · {params.output_format.upper()} · {size} · {quality}\nПример: {example}.{params.output_format}'
