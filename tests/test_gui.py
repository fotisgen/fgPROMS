"""GUI tests with a real browser (Playwright + Chromium).

The fgPROMS server is started in-process on a free port; Chromium drives the
actual frontend: editor, Run button, tabs, results tables, Plotly charts,
language toggle, projects and CSV export.
"""
import socket
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api",
                    reason="Playwright not installed — run test.bat first")

import uvicorn  # noqa: E402
from backend.app import app  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.1)
    yield url
    server.should_exit = True


@pytest.fixture(scope="module")
def page(base_url):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.set_default_timeout(30000)
        pg.goto(base_url)
        # wait for the editor to be initialised and the default example loaded
        pg.wait_for_selector(".CodeMirror-code")
        yield pg
        browser.close()


def _run(page):
    before = int(page.evaluate("document.body.dataset.runCount || '0'"))
    page.click("#btnRun")
    # wait until the ENTIRE run finished (renders + auto tab switch),
    # not just for a stale summary from a previous run
    page.wait_for_function(
        "n => Number(document.body.dataset.runCount || 0) > n", arg=before,
        timeout=60000)


def _wait_results(page):
    page.wait_for_selector("#tab-results .summary", timeout=60000)
    page.wait_for_selector("#tab-results.tab-body.active", timeout=60000)


def test_page_loads(page):
    assert "fgPROMS" in page.title()
    assert page.locator(".CodeMirror").count() == 1
    # default example is preloaded into the editor (async — wait for it)
    page.wait_for_function(
        "() => document.querySelector('.CodeMirror').innerText.includes('MODEL CSTR')")


def test_run_cstr_shows_results(page):
    _run(page)
    _wait_results(page)
    assert "ok" in page.locator("#tab-results .summary").get_attribute("class")
    rows = page.locator("#tab-results table.grid tbody tr").all_text_contents()
    joined = " | ".join(rows)
    assert "Fout" in joined and "1" in joined
    assert "0.333333" in joined      # CA = 1/3
    assert "0.166667" in joined      # r = 1/6


def test_run_tank_produces_plot(page):
    page.select_option("#exampleSel", "tank.gPROMS")
    _run(page)
    _wait_results(page)      # run is async — wait for it before switching tabs
    page.click("[data-tab=plots]")
    page.wait_for_selector("#tab-plots #plotArea svg", timeout=60000)
    # the plot shows at least three traces (h, C, Fout)
    traces = page.locator("#tab-plots #plotArea .scatterlayer .trace").count()
    assert traces >= 3


def test_run_optimization_shows_objective(page):
    page.select_option("#exampleSel", "cstr_optimize.gPROMS")
    _run(page)
    _wait_results(page)
    summary = page.locator("#tab-results .summary").inner_text()
    assert "1.81818" in summary        # analytical optimum


def test_run_flowsheet_shows_diagram(page):
    page.select_option("#exampleSel", "mixer_splitter.gPROMS")
    _run(page)
    _wait_results(page)      # run is async — wait for it before switching tabs
    page.click("[data-tab=model]")
    page.wait_for_selector("#tab-model .flowsheet svg", timeout=60000)
    # two unit boxes labelled MX and SP
    boxes = page.locator("#tab-model .flowsheet svg rect").count()
    assert boxes == 2
    text = page.locator("#tab-model .flowsheet svg").text_content()
    assert "MX" in text and "SP" in text
    # results contain the analytical values
    page.click("[data-tab=results]")
    rows = page.locator("#tab-results table.grid tbody tr").all_text_contents()
    joined = " | ".join(rows)
    assert "1.7" in joined and "0.6" in joined


def test_language_toggle(page):
    page.click("#btnLang")
    assert page.locator("#btnRun").inner_text().strip() == "Run"
    assert "Model" in page.locator(".tab").first.inner_text()
    page.click("#btnLang")
    assert page.locator("#btnRun").inner_text().strip() == "Εκτέλεση"


def test_project_save_and_load(page, base_url):
    name = "gui_test_project.gPROMS"
    page.select_option("#exampleSel", "cstr.gPROMS")
    code_before = page.locator(".CodeMirror").inner_text()

    def accept(dialog):
        dialog.accept(name)
    page.once("dialog", accept)
    page.click("#btnSaveAs")
    page.wait_for_function(
        """() => {
          const sel = document.querySelector('#projectSel');
          return [...sel.options].some(o => o.value === arguments[0]);
        }""".replace("arguments[0]", repr(name)))

    # change the editor, then reload from the project dropdown
    page.evaluate("editor.setValue('MODEL Changed END')")
    page.select_option("#projectSel", name)
    page.wait_for_function(
        "() => document.querySelector('.CodeMirror').innerText.includes('MODEL CSTR')")
    # cleanup: delete the project via the API
    import httpx
    httpx.delete(f"{base_url}/api/projects/{name}", timeout=10)


def test_csv_export_download(page):
    page.select_option("#exampleSel", "tank.gPROMS")
    _run(page)
    page.wait_for_selector("#tab-results #btnCsv", timeout=60000)
    with page.expect_download() as dl_info:
        page.click("#tab-results #btnCsv")
    download = dl_info.value
    assert download.suggested_filename.endswith(".csv")


def test_xy_plot_for_arrays(page):
    """Array results (phase envelope) offer an x-y plot instead of time
    series."""
    page.select_option("#exampleSel", "phase_envelope.gPROMS")
    _run(page)
    page.click("[data-tab=plots]")
    page.wait_for_selector("#tab-plots #plotArea svg", timeout=60000)
    # at least one trace drawn (Tb by default)
    traces = page.locator("#tab-plots #plotArea .scatterlayer .trace").count()
    assert traces >= 1
    # the X selector exists and offers p among its options
    assert page.locator("#xyX option[value='p']").count() == 1


def test_heatmap_for_2d_arrays(page):
    """2D results (conduction field) render as a heatmap."""
    page.select_option("#exampleSel", "conduction_2d.gPROMS")
    _run(page)
    page.click("[data-tab=plots]")
    page.wait_for_selector("#tab-plots #plotArea svg", timeout=60000)
    # the heatmap variable selector exists and offers the field
    assert page.locator("#hmVar option[value='t']").count() == 1
    # a heatmap trace (rendered as an image layer) is present
    assert page.locator("#tab-plots #plotArea image").count() >= 1
