"""Behaviour checks for editing, persistence, resumable exports and watching."""
from dataclasses import replace
from pathlib import Path
import threading
import time
import cv2
import numpy as np
import pytest
from PIL import Image

from raw_engine import RawEngine, ProcessingParams, ProcessingCancelled
from imaging import LocalAdjustment, tonal, geometry, lens_correct, mask_weights
from metadata import exif_bytes, embed_exif
from studio import EditSession, PhotoState, save_preset, load_preset, save_lens_profile, match_lens_profile
from batch import run_batch, FolderWatcher, plan_outputs


def neutral(**kwargs):
    return ProcessingParams(auto_exposure=False,tone_mapping=False,sharpen=False,**kwargs)


def test_tones_target_different_brightness_ranges():
    samples = np.repeat(np.array([.15,.5,.9],np.float32)[None,:,None],3,axis=2)
    shadows = tonal(samples,shadows=.5)-samples
    highlights = samples-tonal(samples,highlights=-.5)
    assert shadows[0,0,0]>shadows[0,2,0]>0
    assert highlights[0,2,0]>highlights[0,0,0]>=0


def test_color_and_monochrome_keep_high_precision():
    image = np.arange(8000,dtype=np.uint16).reshape(40,200)
    rgb = np.stack([image,image+2000,image+4000],axis=2)
    engine = RawEngine()
    changed = engine.apply_corrections(rgb,neutral(temperature=.3,tint=.2,vibrance=.2,saturation=1.2))
    assert changed.dtype==np.float32 and np.isfinite(changed).all()
    assert np.unique(engine.to_uint16(changed)).size>256
    mono = engine.apply_corrections(rgb,neutral(monochrome=True))
    np.testing.assert_allclose(mono[:,:,0],mono[:,:,1])
    np.testing.assert_allclose(mono[:,:,1],mono[:,:,2])


def test_geometry_crop_rotation_and_flips():
    rgb = np.arange(6*8*3,dtype=np.float32).reshape(6,8,3)
    p = neutral(rotation=1,flip_horizontal=True,crop=(.25,0,.75,.5))
    expected = np.rot90(rgb,-1)[:,::-1]
    h,w = expected.shape[:2]
    np.testing.assert_array_equal(geometry(rgb,p),expected[:round(h*.5),round(w*.25):round(w*.75)])


def test_local_masks_follow_normalized_coordinates_at_both_resolutions():
    mask = LocalAdjustment(points=((.25,.5),),radius=.1,exposure=.7)
    engine = RawEngine()
    for h,w in [(120,160),(240,320)]:
        rgb = np.full((h,w,3),12000,np.uint16)
        base = engine.apply_corrections(rgb,neutral())
        result = engine.apply_corrections(rgb,neutral(masks=(mask,)))
        assert result[h//2,w//4,0]>base[h//2,w//4,0]
        np.testing.assert_array_equal(result[h//2,3*w//4],base[h//2,3*w//4])
    gradient = mask_weights((100,200,3),LocalAdjustment(kind='gradient',points=((0,0),(1,0))))
    assert gradient[50,0]==0 and gradient[50,-1]>.99


def test_lens_neutral_profile_is_identity_and_can_cancel():
    rng = np.random.default_rng(4)
    rgb = rng.random((300,400,3),dtype=np.float32)
    np.testing.assert_allclose(lens_correct(rgb,neutral(lens_enabled=True)),rgb,atol=1e-5)
    corrected = lens_correct(rgb,neutral(lens_enabled=True,lens_k1=.1,ca_red=.005,vignette=.1))
    assert not np.allclose(corrected,rgb) and np.isfinite(corrected).all()
    def stop(): raise ProcessingCancelled()
    with pytest.raises(ProcessingCancelled): lens_correct(rgb,neutral(lens_enabled=True),stop)


def test_exact_preview_export_dimensions_and_metadata(dng_path,tmp_path):
    engine = RawEngine()
    params = neutral(rotation=1,crop=(.1,.1,.9,.9),max_edge=160,preserve_exif=True,output_format='png')
    preview = engine.preview(dng_path,params,draft=False)
    assert preview.after.size==preview.dimensions
    assert preview.after.size==preview.before.size
    assert preview.histogram and len(preview.histogram['rgb'])==3
    result = engine.process_file(dng_path,tmp_path/'photo.png',params)
    assert max(result.dimensions)==160
    assert cv2.imread(str(result.output_path),cv2.IMREAD_UNCHANGED).dtype==np.uint16
    with Image.open(result.output_path) as image:
        assert image.getexif()[274]==1
        assert image.getexif().get_ifd(34665)[40962]==result.dimensions[0]


@pytest.mark.parametrize('extension',['jpg','png'])
def test_selected_exif_fields_do_not_change_encoded_pixels(tmp_path,extension):
    params = neutral(output_format=extension,preserve_exif=True,exif_date=False)
    metadata = {'Image Model':'Test Camera','EXIF LensModel':'Test Lens','EXIF DateTimeOriginal':'2026:10:04 12:00:00','EXIF ExposureTime':[1,125]}
    rgb = np.random.default_rng(3).random((30,40,3),dtype=np.float32)
    destination = tmp_path/f'image.{extension}'
    RawEngine().save_image(rgb,destination,params,metadata=metadata)
    with Image.open(destination) as image:
        exif = image.getexif()
        assert exif[272]=='Test Camera'
        sub = exif.get_ifd(34665)
        assert sub[42036]=='Test Lens' and 36867 not in sub
        assert float(sub[33434])==pytest.approx(1/125)
    if extension=='png':
        pixels = cv2.imread(str(destination),cv2.IMREAD_UNCHANGED)[:,:,::-1]
        np.testing.assert_array_equal(pixels,RawEngine.to_uint16(rgb))


def test_project_restores_individual_settings_masks_and_undo(tmp_path):
    session = EditSession()
    paths = [tmp_path/'a.DNG',tmp_path/'b.DNG']
    session.add(paths)
    first = session.photos[str(paths[0])]
    first.set_params(replace(first.params,exposure_ev=1))
    first.set_params(replace(first.params,shadows=.4,masks=(LocalAdjustment(points=((.2,.3),)),)))
    first.undo()
    first.rating=4
    session.photos[str(paths[1])].included=False
    project = tmp_path/'project.rawstudio'
    session.save(project)
    restored = EditSession.load(project)
    photo = restored.photos[str(paths[0])]
    assert photo.params.exposure_ev==1 and photo.params.shadows==0
    assert photo.redo().shadows==.4 and photo.params.masks[0].points==((.2,.3),)
    assert photo.rating==4 and not restored.photos[str(paths[1])].included
    photo.undo()
    photo.set_params(replace(photo.params,exposure_ev=-1))
    assert photo.redo().exposure_ev==-1  # New edit discards the abandoned redo branch.


def test_preset_does_not_copy_crop_masks_or_export(tmp_path):
    source = neutral(exposure_ev=1,shadows=.3,crop=(.1,0,.9,1),max_edge=160)
    target = neutral(crop=(0,.2,1,.8),masks=(LocalAdjustment(points=((.4,.4),)),),output_format='png')
    path = tmp_path/'preset.json'
    save_preset(path,source)
    result = load_preset(path,target)
    assert result.exposure_ev==1 and result.shadows==.3
    assert result.crop==target.crop and result.masks==target.masks and result.output_format=='png'


def test_lens_profile_matches_camera_and_lens(tmp_path):
    meta = {'Image Model':'Camera','EXIF LensModel':'Lens'}
    p = neutral(lens_enabled=True,lens_k1=.08,vignette=.3)
    save_lens_profile(tmp_path/'lens.json',p,meta)
    matched,name = match_lens_profile(tmp_path,meta,neutral())
    assert matched.lens_k1==.08 and matched.lens_enabled and name=='lens.json'
    unmatched,name = match_lens_profile(tmp_path,dict(meta,**{'Image Model':'Other'}),neutral())
    assert name is None and not unmatched.lens_enabled


def test_resume_skips_finished_photos_and_restores_remaining(dng_path,tmp_path):
    second = tmp_path/'second.DNG'
    second.write_bytes(dng_path.read_bytes())
    cancel = threading.Event()
    out = tmp_path/'out'
    per_file = {str(dng_path):neutral(exposure_ev=1),str(second):neutral(exposure_ev=-1)}
    def emit(kind,data):
        if kind=='file_saved': cancel.set()
    first = run_batch(RawEngine(),[dng_path,second],out,neutral(),False,cancel,emit,per_file=per_file)
    assert first.completed==1
    original = (out/'synthetic.jpg').read_bytes()
    mtime = (out/'synthetic.jpg').stat().st_mtime_ns
    second_run = run_batch(RawEngine(),[dng_path,second],out,neutral(),False,threading.Event(),lambda *_:None,per_file=per_file,resume=True)
    assert second_run.succeeded==2 and second_run.failed==0
    assert (out/'synthetic.jpg').stat().st_mtime_ns==mtime
    assert (out/'synthetic.jpg').read_bytes()==original
    assert sorted(p.name for p in out.glob('*.jpg'))==['second.jpg','synthetic.jpg']
    assert (out/'second.jpg').read_bytes()!=original


def test_cancel_while_paused_writes_no_output(dng_path,tmp_path):
    pause,cancel = threading.Event(),threading.Event()
    pause.set()
    result = []
    worker = threading.Thread(target=lambda:result.append(run_batch(RawEngine(),[dng_path],tmp_path/'out',neutral(),False,cancel,lambda *_:None,pause=pause)))
    worker.start()
    cancel.set()
    worker.join(3)
    assert not worker.is_alive() and result[0].cancelled
    assert not list((tmp_path/'out').glob('*.jpg'))


def test_watcher_ignores_existing_and_waits_for_copy_to_stabilize(tmp_path):
    existing=tmp_path/'old.DNG'
    existing.write_bytes(b'old')
    watcher = FolderWatcher(tmp_path)
    assert watcher.poll()==[]
    new = tmp_path/'new.CR2'
    new.write_bytes(b'part')
    assert watcher.poll()==[]
    new.write_bytes(b'part and more')
    assert watcher.poll()==[]
    assert watcher.poll()==[]
    assert watcher.poll()==[new]
    assert watcher.poll()==[]
    new.unlink()
    watcher.poll()
    new.write_bytes(b'new capture')
    assert watcher.poll()==[] and watcher.poll()==[] and watcher.poll()==[new]


def test_filename_templates_and_path_traversal(tmp_path):
    path = tmp_path/'photo.DNG'
    jobs = plan_outputs([path],tmp_path/'out','jpg',template='school_{index:04d}_{stem}')
    assert jobs[0][1].name=='school_0001_photo.jpg'
    for template in ['../{stem}','{stem.__class__}','{unknown}','{index:100000d}','CON/abc']:
        with pytest.raises(ValueError): ProcessingParams(filename_template=template)


@pytest.mark.parametrize('kwargs',[{'crop':(0,0,0,1)},{'masks':({'bad':True},)},{'wb_gains':(0,1,1)},{'max_edge':-1},{'shadows':float('nan')}])
def test_invalid_edits_are_rejected(kwargs):
    with pytest.raises(ValueError): ProcessingParams(**kwargs)


def test_standalone_matches_modular_pipeline(dng_path,tmp_path):
    import RAW_Studio as standalone
    from dataclasses import asdict
    params = neutral(exposure_ev=.3,shadows=.2,temperature=.1,rotation=1,crop=(.1,.1,.9,.9),
                     masks=(LocalAdjustment(kind='gradient',points=((0,0),(1,1))),),max_edge=151,output_format='png',preserve_exif=True)
    modular = RawEngine().process_file(dng_path,tmp_path/'modular.png',params)
    bundled = standalone.RawEngine().process_file(dng_path,tmp_path/'bundled.png',standalone.params_from_dict(asdict(params)))
    assert modular.dimensions==bundled.dimensions
    np.testing.assert_array_equal(cv2.imread(str(modular.output_path),cv2.IMREAD_UNCHANGED),
                                  cv2.imread(str(bundled.output_path),cv2.IMREAD_UNCHANGED))


def test_recovery_snapshot_preserves_origin_and_frozen_edits(tmp_path, dng_path):
    from workflow import session_snapshot, write_snapshot
    from studio import EditSession, read_json
    from raw_engine import ProcessingParams
    session=EditSession();session.add([dng_path]);photo=session.photos[str(dng_path)]
    photo.set_params(replace(photo.params,exposure_ev=.75));photo.rating=4
    session.project_path=tmp_path/'original.rawstudio';session.selected=str(dng_path)
    recovery=tmp_path/'settings'/'recovery.rawstudio'
    snapshot=session_snapshot(session,recovery)
    photo.set_params(replace(photo.params,exposure_ev=-.5))
    write_snapshot(snapshot)
    restored=EditSession.load(recovery)
    assert restored.photos[str(dng_path)].params.exposure_ev==.75
    assert restored.photos[str(dng_path)].rating==4
    assert restored.photos[str(dng_path)].undo().exposure_ev==0
    assert session.project_path==tmp_path/'original.rawstudio'
    assert read_json(recovery)['project_origin']==str(session.project_path)


def test_preview_cache_updates_corrections_and_invalidates_source(dng_path, tmp_path):
    from raw_engine import RawEngine, ProcessingParams
    import os
    class Counted(RawEngine):
        count=0
        def decode(self,*args,**kwargs):
            self.count+=1
            return super().decode(*args,**kwargs)
    engine=Counted();params=ProcessingParams()
    first=engine.preview(dng_path,params)
    second=engine.preview(dng_path,replace(params,exposure_ev=-1))
    assert engine.count==1 and first.after.tobytes()!=second.after.tobytes()
    stat=dng_path.stat();os.utime(dng_path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
    engine.preview(dng_path,params)
    assert engine.count==2
    engine.process_file(dng_path,tmp_path/'output.jpg',params)
    assert engine.count==3


def test_automatic_workflow_preserves_geometry_and_export():
    from workflow import automatic_params, EXPORT_PRESETS
    from raw_engine import ProcessingParams
    original=ProcessingParams(exposure_ev=1.5,temperature=.4,crop=(.1,.1,.9,.9),rotation=1,max_edge=123)
    result=automatic_params(original)
    assert result.auto_exposure and result.tone_mapping and result.exposure_ev==0
    assert result.crop==original.crop and result.rotation==1 and result.max_edge==123
    assert replace(result,**EXPORT_PRESETS['За последваща обработка']).png_bit_depth==16
