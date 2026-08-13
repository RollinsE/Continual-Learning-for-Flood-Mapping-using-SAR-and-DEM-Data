from pathlib import Path


APP_PATH = Path("streamlit_app/app.py")


def test_streamlit_entrypoint_is_present_and_uses_core_deployment_engine():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'from floods.deployment import predict_scene' in source
    assert 'predict_scene(' in source
    assert 'device="cpu"' in source
    assert '/content/drive/' not in source


def test_streamlit_dependencies_are_separate_from_training_requirements():
    requirements = Path('streamlit_app/requirements.txt').read_text(encoding='utf-8')
    assert 'streamlit==1.60.0' in requirements
    assert 'albumentations' not in requirements
    assert 'scipy' not in requirements
    assert 'planetary-computer' not in requirements


def test_streamlit_checkpoint_paths_are_lfs_tracked():
    attrs = Path('.gitattributes').read_text(encoding='utf-8')
    assert 'streamlit_app/deployments/**/assets/checkpoints/*.pth filter=lfs' in attrs

def test_streamlit_checkpoint_is_not_ignored_by_git_rules():
    gitignore = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "!streamlit_app/deployment/**" in gitignore


def test_streamlit_public_errors_do_not_render_tracebacks():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "st.exception(" not in source
    assert "LOG.exception(" in source


def test_streamlit_cleans_previous_workspace_before_new_prediction():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "_clear_previous_workspace()" in source
    assert 'st.session_state["floodmap_work_dir"]' in source



def test_streamlit_supports_model_catalogue_and_collection_inputs():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'APP_DIR / "deployments"' in source
    assert 'accept_multiple_files=True' in source
    assert 'build_collection_mosaic' in source
    assert 'canonical tile identifier' in source.lower()


def test_streamlit_uses_neutral_public_branding():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'page_title="Flood Extent Mapping"' in source
    assert 'MMFlood Flood Mapping' not in source


def test_streamlit_catalogue_has_staging_readme():
    text = Path("streamlit_app/deployments/README.md").read_text(encoding="utf-8")
    assert "floodmap export-deployment" in text
    assert "Git LFS" in text
    assert "No trained checkpoint is included" in text


def test_streamlit_explains_collection_nodata_coverage():
    app_text = Path("streamlit_app/app.py").read_text(encoding="utf-8")
    report_text = Path("streamlit_app/collection.py").read_text(encoding="utf-8")
    assert "grey = no input coverage" in app_text
    assert "grey = no input coverage" in report_text
