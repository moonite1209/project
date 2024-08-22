import os
import dearpygui.dearpygui as dpg
import numpy as np

def main():
    mask_dir = 'data/lerf_ovs/waldo_kitchen/language_features_dim3'
    feature_level = 3
    width = 985
    height = 725
    render_buffer = np.zeros((width, height, 3), dtype=np.float32)

    current = 0
    segments = [os.path.join(mask_dir,p) for p in os.listdir(mask_dir) if p.endswith('s.npy')]
    features = [os.path.join(mask_dir, p) for p in os.listdir(mask_dir) if p.endswith('f.npy')]
    segments.sort()
    features.sort()

    dpg.create_context()
    ### register texture
    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width, height, render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

    ### register window
    with dpg.window(tag="_primary_window", width=width+300, height=height):
        dpg.add_image("_texture")   # add the texture

    dpg.set_primary_window("_primary_window", True)

    def button_callback(sender, app_data, user_data):
        current, offset = user_data
        current+=offset
        print(current)
    with dpg.window(label="Control", tag="_control_window", width=300, height=550, pos=[width+10, 0]):
        dpg.add_text("\noption: ", tag="option")
        dpg.add_text("x: ", tag="x")
        dpg.add_text("y: ", tag="y")
        dpg.add_text('index: ', tag='index')
        dpg.add_button(label='prev', callback=button_callback, user_data=current)
        dpg.add_button(label='next', callback=button_callback, user_data=current)

    def mouse_click_handler(sender, app_data, user_data):
        print(sender, app_data)
        pos = dpg.get_mouse_pos()
        dpg.set_value('x', f'x: {pos[0]}')
        dpg.set_value('y', f'y: {pos[1]}')

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=mouse_click_handler)


    dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=width+320, height=height, resizable=False)

    ### global theme
    with dpg.theme() as theme_no_padding:
        with dpg.theme_component(dpg.mvAll):
            # set all padding to 0 to avoid scroll bar
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
    dpg.bind_item_theme("_primary_window", theme_no_padding)

    dpg.setup_dearpygui()

    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

main()