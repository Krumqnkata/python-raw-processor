"""Versioned projects, presets, per-photo edit history and lens profiles."""
from dataclasses import asdict, fields, replace
from pathlib import Path
import json
import os
import tempfile
from raw_engine import ProcessingParams
from imaging import LocalAdjustment


def atomic_json(path, data):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as handle:
            name = handle.name
            json.dump(data,handle,ensure_ascii=False,indent=2,allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name,path)
        name = None
    finally:
        if name: Path(name).unlink(missing_ok=True)


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 25_000_000:
        raise ValueError('Файлът с настройки е твърде голям.')
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def params_from_dict(data):
    if not isinstance(data,dict) or set(data)-{f.name for f in fields(ProcessingParams)}:
        raise ValueError('Непознати настройки в проекта/профила.')
    data = dict(data)
    data['masks'] = tuple(LocalAdjustment(**m) for m in data.get('masks',()))
    return ProcessingParams(**data)


BUILTIN_PRESETS = {
    'Естествено':{},
    'Портрет':{'tone_strength':.18,'shadows':.18,'vibrance':.12,'sharpen_amount':.25},
    'Пейзаж':{'tone_strength':.38,'highlights':-.2,'vibrance':.25,'sharpen_amount':.6},
    'Черно-бяло':{'monochrome':True,'tone_strength':.4},
}
# Geometry and masks are photo-specific. Export options belong to the job.
EDIT_FIELDS = {f.name for f in fields(ProcessingParams)}-{
    'crop','rotation','straighten','flip_horizontal','flip_vertical','masks',
    'max_edge','filename_template','preserve_exif','exif_camera','exif_date','exif_exposure',
    'output_format','quality','png_compression','png_bit_depth',
}
EXPORT_FIELDS = {f.name for f in fields(ProcessingParams)}-EDIT_FIELDS-{
    'crop','rotation','straighten','flip_horizontal','flip_vertical','masks',
}


class PhotoState:
    def __init__(self, path, params=None, included=True, rating=0):
        self.path = Path(path).expanduser().resolve()
        self.params = params or ProcessingParams()
        self.included = bool(included)
        if type(rating) is not int or not 0 <= rating <= 5:
            raise ValueError('Оценката трябва да е от 0 до 5.')
        self.rating = rating
        self.history = [self.params]
        self.cursor = 0

    def set_params(self, params):
        if self.params == params:
            return
        self.history = self.history[:self.cursor+1]
        self.history.append(params)
        self.history = self.history[-100:]
        self.cursor = len(self.history)-1
        self.params = params

    def undo(self):
        self.cursor = max(0,self.cursor-1)
        self.params = self.history[self.cursor]
        return self.params

    def redo(self):
        self.cursor = min(len(self.history)-1,self.cursor+1)
        self.params = self.history[self.cursor]
        return self.params

    def serialize(self, base):
        try: path = os.path.relpath(self.path,base)
        except ValueError: path = str(self.path)
        return {'path':path,'params':asdict(self.params),'included':self.included,'rating':self.rating,
                'history':[asdict(p) for p in self.history],'cursor':self.cursor}


class EditSession:
    def __init__(self):
        self.photos = {}
        self.defaults = ProcessingParams()
        self.output = ''
        self.selected = ''
        self.project_path = None

    def add(self, paths):
        known = {os.path.normcase(key) for key in self.photos}
        for path in paths:
            key = str(Path(path).expanduser().resolve())
            if os.path.normcase(key) not in known:
                self.photos[key] = PhotoState(path,self.defaults)
                known.add(os.path.normcase(key))

    def save(self, path):
        path = Path(path).expanduser().resolve()
        selected = self.selected
        if selected:
            try: selected = os.path.relpath(selected,path.parent)
            except ValueError: pass
        atomic_json(path,{'version':1,'defaults':asdict(self.defaults),'output':self.output,'selected':selected,
                          'photos':[p.serialize(path.parent) for p in self.photos.values()]})
        self.project_path = path

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        data = read_json(path)
        if not isinstance(data,dict) or data.get('version') != 1 or not isinstance(data.get('photos'),list):
            raise ValueError('Неподдържан проект.')
        session = cls()
        session.defaults = params_from_dict(data.get('defaults',{}))
        session.output = str(data.get('output',''))
        selected = str(data.get('selected',''))
        session.selected = str((path.parent/selected).resolve()) if selected and not Path(selected).is_absolute() else selected
        for row in data['photos']:
            source = Path(row['path'])
            if not source.is_absolute(): source = path.parent/source
            photo = PhotoState(source,params_from_dict(row['params']),row.get('included',True),row.get('rating',0))
            history = [params_from_dict(p) for p in row.get('history',[])][-100:]
            cursor = row.get('cursor',len(history)-1)
            if history and type(cursor) is int and 0 <= cursor < len(history):
                photo.history,photo.cursor = history,cursor
                photo.params = history[cursor]
            session.photos[str(photo.path)] = photo
        session.project_path = path
        return session


def save_preset(path, params):
    atomic_json(path,{'version':1,'settings':{k:v for k,v in asdict(params).items() if k in EDIT_FIELDS}})


def load_preset(path, current):
    data = read_json(path)
    settings = data.get('settings',{})
    if data.get('version') != 1 or set(settings)-EDIT_FIELDS:
        raise ValueError('Неподдържан preset.')
    return replace(current,**settings)


LENS_FIELDS = {'lens_k1','lens_k2','vignette','ca_red','ca_blue'}


def save_lens_profile(path, params, metadata):
    atomic_json(path,{'version':1,'camera':str(metadata.get('Image Model','')),
                     'lens':str(metadata.get('EXIF LensModel','')),
                     'settings':{k:getattr(params,k) for k in LENS_FIELDS}})


def load_lens_profile(path, current):
    data = read_json(path)
    if data.get('version') != 1 or set(data.get('settings',{}))-LENS_FIELDS:
        raise ValueError('Неподдържан профил за обектив.')
    return replace(current,lens_enabled=True,**data['settings'])


def match_lens_profile(folder, metadata, current):
    camera,lens = str(metadata.get('Image Model','')),str(metadata.get('EXIF LensModel',''))
    if not camera or not lens:
        return current,None
    for path in sorted(Path(folder).glob('*.json')):
        try:
            data = read_json(path)
            if data.get('camera')==camera and data.get('lens')==lens:
                return load_lens_profile(path,current),path.name
        except (ValueError,KeyError,TypeError,OSError):
            continue
    return current,None
