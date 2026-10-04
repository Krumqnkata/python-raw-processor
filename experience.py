"""User-facing workflow. All methods and widgets run on the Tk thread."""
from dataclasses import replace
from pathlib import Path
import math
import threading
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from studio import EditSession, BUILTIN_PRESETS, EDIT_FIELDS, EXPORT_FIELDS, atomic_json, read_json
from raw_engine import ProcessingParams, RAW_EXTENSIONS
from batch import discover_raws, unique_inputs
from workflow import (EXPORT_PRESETS, automatic_params, config_directory, load_preferences,
                      session_snapshot, write_snapshot, export_description)

UX_CARD = ('#ffffff','#1a212a')
UX_MUTED = ('#596577','#9eacbd')
UX_ACCENT = ('#087c6d','#20b99b')


class HelpTip:
    def __init__(self, widget, text):
        self.widget,self.text,self.window,self.timer = widget,text,None,None
        widget.bind('<Enter>',self.enter,add='+')
        widget.bind('<Leave>',self.leave,add='+')
        widget.bind('<ButtonPress>',self.leave,add='+')
        widget.bind('<Destroy>',self.leave,add='+')

    def enter(self,event=None):
        self.leave()
        self.timer=self.widget.after(650,self.show)

    def show(self):
        self.timer=None
        if not self.widget.winfo_exists(): return
        self.window=ctk.CTkToplevel(self.widget)
        self.window.overrideredirect(True)
        self.window.geometry(f'+{self.widget.winfo_rootx()+10}+{self.widget.winfo_rooty()+self.widget.winfo_height()+4}')
        ctk.CTkLabel(self.window,text=self.text,wraplength=280,justify='left',fg_color=UX_CARD,
                     corner_radius=8).pack(padx=1,pady=1)

    def leave(self,event=None):
        if self.timer:
            try: self.widget.after_cancel(self.timer)
            except tk.TclError: pass
            self.timer=None
        if self.window:
            try: self.window.destroy()
            except tk.TclError: pass
            self.window=None


class StudioExperience:
    def _init_experience(self):
        self._config_dir=config_directory()
        self._preferences=load_preferences(self._config_dir)
        self.mode_var=tk.StringVar(value=self._preferences.get('mode','Лесен'))
        self.auto_import_var=tk.BooleanVar(value=True)
        self.export_preset_var=tk.StringVar(value='Пълна резолюция')
        self.copy_group_var=tk.StringVar(value='Всички общи корекции')
        self._simple_sliders={}
        self._export_buttons=[]
        self._selected_photos=set()
        self._last_selected_index=None
        self._export_pending=False
        self._queued_drop=[]
        self._export_override=None
        self._failed_files=[]
        self._displayed_path=None
        self._autosave_worker=None
        self._autosave_timer=None
        self._saved_snapshot=None
        self._autosave_path=self._config_dir/'recovery.rawstudio'
        self._recovery_available=self._autosave_path.is_file()
        self._sidebar_visible=True
        self._details_visible=False
        self._tips=[]
        self._ux_ready=False
        self._recovery_snapshot=None
        self._autosave_next=None
        self._gallery_tips=[]

    def _accordion(self,parent,title,opened=True):
        frame=ctk.CTkFrame(parent,fg_color='transparent')
        frame.pack(fill='x',pady=(10,0))
        body=ctk.CTkFrame(frame,fg_color='transparent')
        button=ctk.CTkButton(frame,text='▾  '+title,anchor='w',fg_color='transparent',
                            text_color=UX_MUTED,height=30)
        button.pack(fill='x',padx=8)
        body.pack(fill='x')
        def toggle():
            if body.winfo_manager(): body.pack_forget(); button.configure(text='▸  '+title)
            else: body.pack(fill='x'); button.configure(text='▾  '+title)
        button.configure(command=toggle)
        if not opened:body.pack_forget();button.configure(text='▸  '+title)
        return body

    def _build_simple_sidebar(self):
        self.simple_sidebar=ctk.CTkFrame(self,width=310,corner_radius=0,fg_color=UX_CARD)
        self.simple_sidebar.grid(row=0,column=0,sticky='nsew')
        self.simple_sidebar.grid_propagate(False)
        self.simple_sidebar.grid_columnconfigure(0,weight=1)
        self.simple_sidebar.grid_rowconfigure(1,weight=1)
        ctk.CTkLabel(self.simple_sidebar,text='RAW Studio',font=ctk.CTkFont(size=26,weight='bold')).grid(row=0,column=0,pady=14)
        panel=ctk.CTkScrollableFrame(self.simple_sidebar,fg_color=UX_CARD)
        panel.grid(row=1,column=0,sticky='nsew',padx=5)
        self._label(panel,'1  ДОБАВИ СНИМКИТЕ',text_color=UX_MUTED).pack(fill='x',padx=12,pady=(8,6))
        self._button(panel,'+ Добави снимки',self._choose_files,fg_color=UX_ACCENT)
        self._button(panel,'Добави папка',self._choose_folder)
        self._switch(panel,'Автоматично подобрение',self.auto_import_var,command=lambda:None,setting=False)
        self._label(panel,'Светлина, цвят и детайл се коригират автоматично. Оригиналите се запазват.',wraplength=245,justify='left',text_color=UX_MUTED).pack(fill='x',padx=12,pady=4)
        self._label(panel,'2  ПРЕГЛЕДАЙ РЕЗУЛТАТА',text_color=UX_MUTED).pack(fill='x',padx=12,pady=(18,6))
        menu=ctk.CTkOptionMenu(panel,values=list(BUILTIN_PRESETS),variable=self.preset_var,
                             command=lambda name:self._ui_action(self._apply_easy_style,name))
        menu.pack(fill='x',padx=12,pady=6)
        self._setting_widgets.append(menu)
        manual=self._accordion(panel,'КОРЕКЦИИ ПО ЖЕЛАНИЕ',opened=False)
        self._label(manual,'За текущата снимка',text_color=UX_MUTED).pack(fill='x',padx=12)
        for key,title,low,high,initial in [('exposure_ev','Яркост',-3,3,0),('shadows','Сенки',-1,1,0),
                                         ('temperature','Топлина на цветовете',-1,1,0),('vibrance','Живост на цветовете',-1,1,0)]:
            row=ctk.CTkFrame(manual,fg_color='transparent');row.pack(fill='x',padx=12,pady=(8,0))
            label=self._label(row,title);label.pack(side='left')
            value=self._label(row,'0.00',text_color=UX_MUTED);value.pack(side='right')
            def changed(v,k=key): self._simple_changed(k,v)
            slider=ctk.CTkSlider(manual,from_=low,to=high,number_of_steps=120,command=changed,progress_color=UX_ACCENT)
            slider.set(initial);slider.pack(fill='x',padx=12,pady=5)
            self._simple_sliders[key]=(slider,value)
            self._setting_widgets.append(slider)
            self._slider_labels[slider]=(value,lambda v:f'{v:+.2f}')
            self._decorate_slider(slider,value,label,title,initial,changed)
        self._button(manual,'Автоматично подобри серията',self._auto_all)
        self._label(panel,'3  ЗАПАЗИ ГОТОВИТЕ СНИМКИ',text_color=UX_MUTED).pack(fill='x',padx=12,pady=(18,6))
        presets=ctk.CTkOptionMenu(panel,values=list(EXPORT_PRESETS),variable=self.export_preset_var,command=self._apply_export_preset)
        presets.pack(fill='x',padx=12,pady=6);self._setting_widgets.append(presets)
        self.simple_output_entry=ctk.CTkEntry(panel,textvariable=self.output_var,placeholder_text='Папка за резултатите')
        self.simple_output_entry.pack(fill='x',padx=12,pady=6);self._locked_widgets.append(self.simple_output_entry)
        self._button(panel,'Избери папка за резултатите',self._choose_output)
        self.export_summary=self._label(panel,'Добави снимки, за да започнеш.',wraplength=245,justify='left',text_color=UX_MUTED)
        self.export_summary.pack(fill='x',padx=12,pady=8)
        self.simple_export_error=self._label(panel,'',wraplength=245,justify='left',text_color='#ed7878')
        self.simple_export_error.pack(fill='x',padx=12)
        self.autosave_label=self._label(panel,'Редакциите се запазват автоматично.',text_color=UX_MUTED,wraplength=245)
        self.autosave_label.pack(fill='x',padx=12,pady=8)
        footer=ctk.CTkFrame(self.simple_sidebar,fg_color='transparent');footer.grid(row=2,column=0,sticky='ew',padx=16,pady=12)
        self.simple_export_button=ctk.CTkButton(footer,text='Обработи и запази',height=42,fg_color=UX_ACCENT,command=self._request_export)
        self.simple_export_button.pack(fill='x');self._export_buttons.append(self.simple_export_button)
        ctk.CTkButton(footer,text='Разширен режим',fg_color='transparent',text_color=UX_MUTED,
                      command=lambda:self._set_mode('Разширен')).pack(fill='x',pady=(8,0))

    def _finish_experience(self):
        self._ux_ready=True
        self._set_mode(self.mode_var.get(),save=False)
        self._sync_simple_controls()
        self._toggle_details(False)
        self._setup_drop()
        for variable in [self.output_var,self.max_edge_var,self.template_var,self.format_var]:
            variable.trace_add('write',lambda *_:self._update_experience())
        self.bind('<F6>',self._sidebar_key,add='+')
        self.bind('<Control-o>',lambda e:self._ui_action(self._choose_files))
        self.bind('<Control-e>',lambda e:self._request_export())
        self.after(150,self._offer_recovery)
        self._autosave_timer=self.after(10000,self._autosave_tick)
        self.main_panel.bind("<Configure>",self._responsive_layout,add="+")
        self._update_experience()

    def _responsive_layout(self,event=None):
        simple=self.mode_var.get()=='Лесен'
        for widget in [self.file_menu,self.preview_button,self.preview_switch]:
            widget.grid_remove() if simple else widget.grid()
        short=self.main_panel.winfo_height()/self.main_panel._get_widget_scaling()<700
        if short!=getattr(self,'_compact_height',None):
            self._compact_height=short
            if short:
                self.gallery_actions.pack_forget();self.tool_hint.grid_remove()
            else:
                self.gallery.pack_forget();self.gallery_actions.pack(fill='x',padx=8,pady=2)
                self.gallery.pack(fill='x',padx=4,pady=2);self.tool_hint.grid()
        compact=self.main_panel.winfo_width()/self.main_panel._get_widget_scaling()<650
        self.compare_control.grid_configure(row=0 if simple else 2 if compact else 1,column=0 if simple or compact else 1,columnspan=2 if simple or compact else 1,sticky="ew" if simple or compact else "")

    def _set_mode(self,mode,save=True):
        if mode not in {'Лесен','Разширен'}: mode='Лесен'
        self.mode_var.set(mode)
        self.simple_sidebar.grid_remove();self.advanced_sidebar.grid_remove()
        if self._sidebar_visible:
            (self.simple_sidebar if mode=='Лесен' else self.advanced_sidebar).grid()
        self._sync_simple_controls()
        if self._ux_ready:self._responsive_layout()
        if save:self._save_preferences()

    def _build_workspace_header(self,header):
        actions=ctk.CTkFrame(header,fg_color='transparent');actions.pack(side='right')
        for title,action in [('Настройки',self._toggle_sidebar),('Подробности',lambda:self._toggle_details(not self._details_visible)),('Проект',self._project_actions)]:
            ctk.CTkButton(actions,text=title,width=85,height=26,command=action).pack(side='left',padx=2)

    def _project_actions(self):
        menu=tk.Menu(self,tearoff=False)
        state='disabled' if self._busy else 'normal'
        menu.add_command(label='Отвори проект…',state=state,command=lambda:self._ui_action(self._load_project))
        menu.add_command(label='Запази проект…',state=state,command=lambda:self._ui_action(self._save_project))
        menu.tk_popup(self.winfo_pointerx(),self.winfo_pointery())
        menu.grab_release()
        self._project_popup=menu

    def _toggle_sidebar(self):
        self._sidebar_visible=not self._sidebar_visible
        self._set_mode(self.mode_var.get(),save=False)

    def _sidebar_key(self,event):
        focused=self.focus_get()
        if focused and focused.winfo_class() in {'Entry','Text'}:return
        self._toggle_sidebar();return 'break'

    def _toggle_details(self,visible):
        self._details_visible=visible
        for widget in [self.log_header,self.log_box]:
            widget.grid() if visible else widget.grid_remove()

    def _build_welcome_and_report(self,main):
        self.welcome=ctk.CTkFrame(self.preview_card,fg_color=UX_CARD,corner_radius=12)
        self.welcome.grid(row=0,column=0,columnspan=2,sticky='nsew')
        self.welcome.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(self.welcome,text='Добави снимки. Ние подготвяме резултата.',font=ctk.CTkFont(size=23,weight='bold'),wraplength=450).pack(pady=(35,10))
        self.drop_hint=ctk.CTkLabel(self.welcome,text='Пусни RAW файлове или папка тук',font=ctk.CTkFont(size=16))
        self.drop_hint.pack(pady=10)
        ctk.CTkButton(self.welcome,text='+ Добави снимки',fg_color=UX_ACCENT,command=lambda:self._ui_action(self._choose_files)).pack(pady=5)
        ctk.CTkButton(self.welcome,text='Добави папка',command=lambda:self._ui_action(self._choose_folder)).pack(pady=5)
        self.recent_frame=ctk.CTkFrame(self.welcome,fg_color='transparent');self.recent_frame.pack(fill='x',padx=35,pady=15)
        self._refresh_recents()
        self.tool_hint=self._label(main,'Колелце: увеличение · F6: скрий/покажи настройките',text_color=UX_MUTED,wraplength=600)
        self.tool_hint.grid(row=9,column=0,sticky='ew',pady=4)
        self.report_card=ctk.CTkFrame(main,fg_color=UX_CARD)
        self.report_card.grid(row=10,column=0,sticky='ew',pady=5)
        self.report_label=self._label(self.report_card,'');self.report_label.pack(side='left',padx=12,pady=8)
        ctk.CTkButton(self.report_card,text='Отвори папката',width=130,command=self._open_output).pack(side='right',padx=5)
        self.retry_button=ctk.CTkButton(self.report_card,text='Повтори неуспешните',width=155,command=self._retry_failed)
        self.retry_button.pack(side='right',padx=5)
        self.report_card.grid_remove()
        self.recovery_bar=ctk.CTkFrame(main,fg_color=UX_CARD)
        self.recovery_bar.grid(row=11,column=0,sticky='ew',pady=5)
        self._label(self.recovery_bar,'Има запазена предишна сесия.').pack(side='left',padx=12)
        ctk.CTkButton(self.recovery_bar,text='Възстанови',width=100,command=self._restore_recovery).pack(side='right',padx=5,pady=8)
        ctk.CTkButton(self.recovery_bar,text='Нова сесия',width=100,command=self._discard_recovery).pack(side='right',padx=5)
        self.recovery_bar.grid_remove()

    def _target_slider(self,key):
        return self.exposure_slider if key=='exposure_ev' else self._extra_sliders[key]

    def _simple_changed(self,key,value):
        target=self._target_slider(key)
        steps=target.cget("number_of_steps");target.configure(number_of_steps=None);target.set(value);target.configure(number_of_steps=steps)
        label,formatter=self._slider_labels[target];label.configure(text=formatter(target.get()))
        slider,label=self._simple_sliders[key];label.configure(text=f'{target.get():+.2f}')
        self._schedule_preview()

    def _sync_simple_controls(self):
        if not hasattr(self,'exposure_slider'):return
        for key,(slider,label) in self._simple_sliders.items():
            value=self._target_slider(key).get()
            steps=slider.cget('number_of_steps');slider.configure(number_of_steps=None);slider.set(value);slider.configure(number_of_steps=steps)
            label.configure(text=f'{value:+.2f}')

    def _decorate_slider(self,slider,value_label,title_label,title,initial,changed):
        def reset(event=None):
            if slider.cget('state')=='disabled':return
            slider.set(initial);changed(slider.get());return 'break'
        slider.bind('<Double-Button-1>',reset,add='+')
        value_label.bind('<Button-1>',lambda e:self._edit_numeric(slider,value_label,changed,title),add='+')
        hints={'Сенки':'Изсветлява или потъмнява тъмните области.',
               'Яркост':'Променя експозицията. 0 запазва автоматичната корекция.',
               'Топлина на цветовете':'Наляво: по-студени цветове. Надясно: по-топли.',
               'Живост на цветовете':'Подсилва предимно по-слабо наситените цветове.'}
        self._tips.append(HelpTip(title_label,hints.get(title,title)+ '\nДвойно щракване: нулиране. Щракни стойността за точно число.'))

    def _edit_numeric(self,slider,label,changed,title):
        if slider.cget('state')=='disabled':return
        row=label.master
        entry=ctk.CTkEntry(row,width=95)
        entry.place(relx=1,rely=.5,anchor='e')
        entry.insert(0,f'{slider.get():.6g}');entry.after(20,entry.focus_set);entry.select_range(0,'end')
        def apply(event=None):
            if not entry.winfo_exists():return
            try:
                value=float(entry.get().replace(',','.'))
                if not math.isfinite(value) or not slider.cget('from_')<=value<=slider.cget('to'):raise ValueError()
            except ValueError:
                entry.configure(border_color='#ed7878')
                self.tool_hint.configure(text=f'{title}: въведи число от {slider.cget("from_")} до {slider.cget("to")}.')
                return 'break'
            steps=slider.cget('number_of_steps');slider.configure(number_of_steps=None);slider.set(value);slider.configure(number_of_steps=steps)
            changed(slider.get());entry.destroy();return 'break'
        entry.bind('<Return>',apply);entry.bind('<Escape>',lambda e:entry.destroy())
        entry.bind('<FocusOut>',apply)

    def _import_defaults(self,current):
        clean=replace(ProcessingParams(),**{k:getattr(current,k) for k in EDIT_FIELDS|EXPORT_FIELDS})
        return replace(automatic_params(clean),**BUILTIN_PRESETS.get(self.preset_var.get(),{})) if self.auto_import_var.get() else clean

    def _apply_easy_style(self,name):
        if self._busy in {'batch','scan'}:return
        self._commit_current()
        preset=ProcessingParams(**BUILTIN_PRESETS[name])
        edits={k:getattr(preset,k) for k in EDIT_FIELDS}
        for photo in self.session.photos.values():
            if photo.included:photo.set_params(replace(photo.params,**edits))
        current=self.session.photos[self._active_path].params if self._active_path in self.session.photos else replace(self._params(),**edits)
        self._load_params(current);self._project_dirty=True;self._schedule_preview()

    def _auto_all(self):
        if self._busy in {'batch','scan'}:return
        self._commit_current()
        for photo in self.session.photos.values():
            if photo.included:photo.set_params(automatic_params(photo.params))
        self.preset_var.set('Естествено')
        params=automatic_params(self._params())
        if self._active_path in self.session.photos:params=self.session.photos[self._active_path].params
        self._load_params(params);self._project_dirty=True;self._schedule_preview();self._update_experience()
        self._log('Автоматични корекции за всички включени снимки. Готови за запазване.')

    def _apply_export_preset(self,name):
        try:
            params=replace(self._params(),**EXPORT_PRESETS[name])
            self._load_params(params);self._project_dirty=True;self._update_experience()
        except ValueError as error:self._show_export_error(str(error))

    def _update_experience(self):
        if not self._ux_ready or self._closing:return
        if self.files:self.welcome.grid_remove()
        else:self.welcome.grid()
        count=sum(p.included for p in self.session.photos.values())
        for button in self._export_buttons:button.configure(text=f'Обработи и запази {count} снимки')
        try:
            p=self._params()
            self.export_summary.configure(text=export_description(p,count,self._active_path))
            self.simple_export_error.configure(text='');self.export_error.configure(text='')
            for entry in [self.edge_entry,self.naming_entry]:entry.configure(border_color=('#979da2','#565b5e'))
            matching=next((name for name,values in EXPORT_PRESETS.items() if all(getattr(p,k)==v for k,v in values.items())),None)
            self.export_preset_var.set(matching or 'Собствени настройки')
        except ValueError as error:self._show_export_error(str(error))
        self._refresh_experience_enabled()

    def _show_export_error(self,message):
        self.simple_export_error.configure(text=message);self.export_error.configure(text=message)
        field=self.edge_entry if 'страна' in message or 'число' in message else self.naming_entry
        field.configure(border_color='#ed7878')

    def _refresh_experience_enabled(self):
        if not self._ux_ready:return
        enabled=bool(self.files) and any(p.included for p in self.session.photos.values()) and self._busy not in {'batch','scan'} and not self._closing and not self._export_pending
        for button in self._export_buttons:button.configure(state='normal' if enabled else 'disabled')
        # Photo selection remains available while the previous preview is cancelled.
        if self._busy=='preview':self.file_menu.configure(state='normal')
        self.progress_card.grid() if self._busy in {'batch','scan'} else self.progress_card.grid_remove()

    def _request_export(self,files_override=None):
        if self._closing or self._busy in {'batch','scan'}:return
        if not self.files:self.simple_export_error.configure(text='Първо добави снимки.');return
        try:self._params()
        except ValueError as error:self._show_export_error(str(error));return
        if not self.output_var.get().strip():
            self.simple_export_error.configure(text='Избери папка за резултатите.');return
        directory=Path(self.output_var.get()).expanduser()
        if directory.exists() and not directory.is_dir():
            self.simple_export_error.configure(text='Пътят за резултатите трябва да е папка.');return
        if self._busy=='preview':
            self._export_pending=True;self._export_override=files_override;self._pending_preview=False;self._cancel.set()
            self.status_label.configure(text='Подготвяне на експорта…');self._refresh_experience_enabled();return
        self._start_batch(files_override=files_override)

    def _show_report(self,summary):
        self.report_label.configure(text=f'{summary.succeeded} готови · {summary.failed} неуспешни'+(' · прекъснато' if summary.cancelled else ''))
        self.retry_button.configure(state='normal' if self._failed_files else 'disabled')
        self.report_card.grid()

    def _retry_failed(self):
        paths=list(self._failed_files)
        if paths:self._request_export(files_override=paths)

    def _setup_drop(self):
        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES
            self.TkdndVersion=TkinterDnD._require(self)
            for widget in [self.welcome,self.preview_card,self.before_image_label._label,self.after_image_label._label]:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind('<<Drop>>',self._drop_files)
            self.drop_hint.configure(text='Пусни RAW файлове или папка тук')
            self._drop_available=True
        except (ImportError,tk.TclError,RuntimeError) as error:
            self._drop_available=False
            self.drop_hint.configure(text='Избери снимки или папка с бутоните по-долу')
            self._log('Влаченето на файлове е недостъпно: '+str(error))

    def _drop_files(self,event):
        if self._busy in {'batch','scan'}:return 'none'
        paths=[Path(p) for p in self.tk.splitlist(event.data)]
        return self._import_paths(paths)

    def _import_paths(self,paths):
        if self._busy=='preview':
            self._queued_drop.extend(paths);self._cancel.set();self._pending_preview=False
            return 'copy'
        recursive=self.recursive_var.get()
        def scan(emit,cancel):
            files=[];errors=[]
            try:
                for path in paths:
                    if cancel.is_set():break
                    if path.is_dir():
                        found,problems=discover_raws(path,recursive,cancel);files.extend(found);errors.extend(problems)
                    elif path.suffix.lower() in RAW_EXTENSIONS:files.append(path)
                    else:errors.append('Неподдържан файл: '+path.name)
                emit('scan_result',{'files':unique_inputs(files),'errors':errors})
            except Exception as error:emit('fatal_error',{'error':str(error)})
            finally:emit('job_done',{})
        self._launch('scan',scan);return 'copy'

    def _build_experience_gallery(self,parent):
        outer=ctk.CTkFrame(parent,fg_color=UX_CARD,corner_radius=8)
        outer.grid(row=7,column=0,sticky='ew',pady=(8,0))
        top=ctk.CTkFrame(outer,fg_color='transparent');top.pack(fill='x',padx=8,pady=2)
        for text,delta in [('◀',-1),('▶',1)]:
            ctk.CTkButton(top,text=text,width=30,height=24,command=lambda d=delta:self._gallery_step(d)).pack(side='left',padx=2)
        self.gallery_label=self._label(top,'Галерия',text_color=UX_MUTED,height=20,font=ctk.CTkFont(size=11));self.gallery_label.pack(side='left',padx=5)
        for title,action in [('Всички',lambda:self._set_all_included(True)),('Нито една',lambda:self._set_all_included(False)),('Текущата',self._include_current_only)]:
            ctk.CTkButton(top,text=title,width=60,height=20,font=ctk.CTkFont(size=10),command=action).pack(side='left',padx=2)
        self.favorite_checkbox=ctk.CTkCheckBox(top,text='Любими',variable=self.favorite_filter_var,command=lambda:self._refresh_gallery(reset=True),width=70,height=20,checkbox_width=14,checkbox_height=14,font=ctk.CTkFont(size=11))
        self.favorite_checkbox.pack(side='right')
        self.gallery_actions=actions=ctk.CTkFrame(outer,fg_color='transparent');actions.pack(fill='x',padx=8,pady=2)
        self.selection_label=self._label(actions,'Щракване: текуща · Ctrl/Shift: избор',text_color=UX_MUTED,font=ctk.CTkFont(size=11),height=20)
        self.selection_label.pack(side='left')
        ctk.CTkButton(actions,text='Включи избраните',width=120,height=23,command=lambda:self._include_selected(True)).pack(side='right',padx=3)
        ctk.CTkButton(actions,text='Изключи избраните',width=125,height=23,command=lambda:self._include_selected(False)).pack(side='right',padx=3)
        self.gallery=ctk.CTkScrollableFrame(outer,orientation='horizontal',height=120,fg_color='transparent')
        self.gallery.pack(fill='x',padx=4,pady=2)

    def _populate_gallery(self,reset):
        if reset:self._gallery_page=0
        self._thumb_cancel.set();self._thumb_epoch+=1
        for tip in self._gallery_tips:tip.leave()
        self._gallery_tips=[]
        self._gallery_buttons.clear();self._gallery_controls.clear()
        for widget in self.gallery.winfo_children():widget.destroy()
        photos=[p for p in self.session.photos.values() if not self.favorite_filter_var.get() or p.rating>0]
        self._selected_photos.intersection_update(str(p.path) for p in photos)
        pages=max(1,math.ceil(len(photos)/40));self._gallery_page=min(self._gallery_page,pages-1)
        page=photos[self._gallery_page*40:(self._gallery_page+1)*40]
        included=sum(p.included for p in self.session.photos.values())
        self.gallery_label.configure(text=f'{len(photos)} · {included} за експорт · {self._gallery_page+1}/{pages}')
        for photo in page:
            key=str(photo.path);active=key==self._active_path;chosen=key in self._selected_photos
            cell=ctk.CTkFrame(self.gallery,width=130,fg_color=('#e0f4ed','#23443d') if chosen else 'transparent',
                            border_width=2 if active else 0,border_color=UX_ACCENT)
            cell.pack(side='left',padx=3,pady=2)
            thumb=self._thumb_cache.get(key)
            image=ctk.CTkImage(light_image=thumb,dark_image=thumb,size=thumb.size) if thumb else None
            button=ctk.CTkButton(cell,text=photo.path.name[:18]+(' •' if photo.cursor else ''),width=120,height=48,image=image,
                                compound='top',font=ctk.CTkFont(size=11),command=lambda p=photo.path:self._gallery_click(p))
            button._thumbnail=image;button.pack(fill='x',padx=2,pady=2)
            button.bind('<Button-3>',lambda e,p=photo.path:self._gallery_context(e,p),add='+')
            button.bind('<ButtonPress-1>',lambda e,p=photo.path:self._capture_gallery_modifiers(e,p),add='+')
            self._gallery_buttons[key]=button
            self._gallery_tips.append(HelpTip(button,photo.path.name+'\nCtrl/Shift: избор · Десен бутон: групови действия\n'+('Има редакции' if photo.cursor else 'Автоматични настройки')))
            included_var=tk.BooleanVar(value=photo.included)
            check=ctk.CTkCheckBox(cell,text='За експорт',height=18,width=110,checkbox_width=15,checkbox_height=15,
                                variable=included_var,command=lambda p=photo,v=included_var:self._set_included(p,v.get()))
            check.pack(padx=4,pady=2)
            stars=ctk.CTkFrame(cell,fg_color='transparent');stars.pack(pady=2)
            for rating in range(1,6):
                star=ctk.CTkButton(stars,text='★' if rating<=photo.rating else '☆',width=22,height=20,
                                  fg_color='transparent',text_color=('#9b7100','#f4cf66'),
                                  command=lambda p=photo,r=rating:self._rating_clicked(p,r))
                star.pack(side='left');self._gallery_controls.append(star)
            self._gallery_controls.extend([button,check])
        self._selection_count();self._update_experience();self._refresh_enabled()
        return page

    def _gallery_context(self,event,path):
        if self._busy in {'batch','scan'}:return
        if str(path) not in self._selected_photos:self._selected_photos={str(path)};self._refresh_gallery()
        menu=tk.Menu(self,tearoff=False)
        menu.add_command(label='Включи избраните за експорт',command=lambda:self._include_selected(True))
        menu.add_command(label='Изключи избраните от експорта',command=lambda:self._include_selected(False))
        menu.tk_popup(event.x_root,event.y_root);menu.grab_release();self._gallery_popup=menu

    def _capture_gallery_modifiers(self,event,path):
        self._gallery_modifiers=(str(path),bool(event.state&4),bool(event.state&1))

    def _gallery_click(self,path,ctrl=None,shift=None):
        if self._busy in {'batch','scan'}:return
        if ctrl is None:
            captured=getattr(self,'_gallery_modifiers',None)
            ctrl,shift=(captured[1:] if captured and captured[0]==str(path) else (False,False))
            self._gallery_modifiers=None
        key=str(path);keys=[str(p.path) for p in self.session.photos.values() if not self.favorite_filter_var.get() or p.rating>0];index=keys.index(key)
        anchor=getattr(self,'_selection_anchor',None)
        self._last_selected_index=keys.index(anchor) if anchor in keys else keys.index(self._active_path) if self._active_path in keys else index
        if shift and self._last_selected_index is not None:
            a,b=sorted([index,self._last_selected_index]);self._selected_photos.update(keys[a:b+1])
        elif ctrl:
            if key in self._selected_photos:self._selected_photos.remove(key)
            else:self._selected_photos.add(key)
        else:self._selected_photos={key}
        self._last_selected_index=index
        self._selection_anchor=key
        self._ui_action(self._select_photo,path);self._refresh_gallery()

    def _selection_count(self):
        self.selection_label.configure(text=f'{len(self._selected_photos)} избрани · Ctrl/Shift: избор')

    def _include_selected(self,value):
        if self._busy in {'batch','scan'}:return
        for key in self._selected_photos:
            if key in self.session.photos:self.session.photos[key].included=value
        self._project_dirty=True;self._refresh_gallery()

    def _rating_clicked(self,photo,rating):
        if self._busy in {'batch','scan'}:return
        self._set_rating(photo,0 if photo.rating==rating else rating);self._refresh_gallery()

    def _save_preferences(self):
        try:
            self._preferences.update(version=1,mode=self.mode_var.get())
            atomic_json(self._config_dir/'preferences.json',self._preferences)
        except (OSError,ValueError) as error:self._log('Настройките не могат да се запазят: '+str(error),'error')

    def _remember_project(self,path):
        path=str(Path(path).resolve())
        old=self._preferences.get('recent',[])
        if not isinstance(old,list):old=[]
        self._preferences['recent']=[path]+[p for p in old if p!=path][:7]
        self._save_preferences();self._refresh_recents()

    def _refresh_recents(self):
        for widget in self.recent_frame.winfo_children():widget.destroy()
        recent=self._preferences.get('recent',[])
        if isinstance(recent,list):
            for path in recent[:4]:
                if isinstance(path,str):ctk.CTkButton(self.recent_frame,text='Проект: '+Path(path).name,
                       fg_color='transparent',text_color=UX_MUTED,command=lambda p=path:self._open_project_path(p)).pack(pady=2)

    def _open_project_path(self,path,recovery=False):
        if self._busy:return
        try:session=EditSession.load(path)
        except (OSError,ValueError,TypeError,KeyError) as error:
            self.simple_export_error.configure(text='Проектът не може да се отвори: '+str(error));return
        self._clear_files();self.session=session;self.output_var.set(session.output);self._load_params(session.defaults)
        self._add_files([p.path for p in session.photos.values()])
        selected=next((key for key,value in self._file_choices.items() if str(value)==session.selected),None)
        if selected:self.selected_var.set(selected);self._active_path=None;self._selection_changed()
        if recovery:
            origin=read_json(path).get('project_origin','');self.session.project_path=Path(origin) if origin else None
        else:self._remember_project(path)
        self._project_dirty=recovery
        self.project_label.configure(text=Path(self.session.project_path).name if self.session.project_path else 'Възстановена сесия')
        self._log('Сесията е възстановена.');self._update_experience()

    def _offer_recovery(self):
        if not self._recovery_available:return
        try:
            data=read_json(self._autosave_path)
            if data.get('photos'):self.recovery_bar.grid()
            else:self._recovery_available=False
        except (OSError,ValueError,AttributeError):self._recovery_available=False

    def _restore_recovery(self):
        if self._busy:return
        self._open_project_path(self._autosave_path,recovery=True)
        self._recovery_available=False;self.recovery_bar.grid_remove()

    def _discard_recovery(self):
        self._recovery_available=False;self.recovery_bar.grid_remove();self._saved_snapshot=None
        self._save_recovery()

    def _autosave_tick(self):
        self._autosave_timer=None
        if self._closing:return
        self._save_recovery()
        self._autosave_timer=self.after(10000,self._autosave_tick)

    def _save_recovery(self):
        if self._recovery_available:return
        try:self._commit_current()
        except ValueError:return
        snapshot=session_snapshot(self.session,self._autosave_path)
        if snapshot==self._saved_snapshot:return
        if self._autosave_worker and self._autosave_worker.is_alive():
            self._autosave_next=snapshot
            return
        self._write_recovery_snapshot(snapshot)

    def _write_recovery_snapshot(self,snapshot):
        def save():
            try:
                write_snapshot(snapshot)
                self.events.put((-1,'autosaved',{'snapshot':snapshot}))
            except (OSError,ValueError) as error:self.events.put((-1,'autosave_error',{'error':str(error)}))
        self._autosave_worker=threading.Thread(target=save,name='studio-recovery',daemon=False)
        self._autosave_worker.start()
