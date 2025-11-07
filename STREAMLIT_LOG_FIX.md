# Streamlit Log Persistence Fix

## Problem

When you click "Back" in Streamlit, the logs disappear because:
1. Streamlit reruns the entire script
2. Local variables (like `log_lines`) are cleared
3. The logs were only stored in local variables, not session state

## Solution Applied

**Changed logs to persist in `st.session_state`**:

- Logs are now saved to `st.session_state.run_logs` as they arrive
- When you click "Back" and return, previous logs are shown
- Logs survive Streamlit reruns and page navigation

## Also Fixed

**Immediate crash detection**:

- If the backend crashes immediately (like `ModuleNotFoundError: No module named 'cv2'`)
- Error output is now captured with `proc.stdout.read()`
- Error is displayed even if the process exits before the log loop runs

## Test It

1. Run Streamlit: `streamlit run streamlit_app.py`
2. Upload PDF and run analysis
3. While running (or after crash), view logs in "📋 Detailed Logs"
4. Click "← Back" button
5. Click "▶️ Run" again
6. **Logs should still be visible** in the expander

## What You'll Now See

If cv2 is not installed, you'll now see:

```
Traceback (most recent call last):
  File "ai_pdf_panel_duplicate_check_AUTO.py", line 16, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
```

**This makes debugging much easier!**

## Still Need to Install Dependencies

The logs will now show the error, but you still need to fix it:

```bash
pip3 install opencv-python-headless scikit-image torch open-clip-torch
```
