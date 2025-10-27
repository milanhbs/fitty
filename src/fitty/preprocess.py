import numpy as np

def trim_growth_curve(time, od, 
                      lag_deriv_threshold=0.002,
                      trim_od_threshold=0.0, 
                      min_points_before_lag=0,
                      post_peak_points=0,
                      smooth_window=3):
    """
    Trim lag and post-peak OD decline.
    
    Parameters
    ----------
    time : array
        Time points (must be sorted).
    od : array
        Optical density values.
    lag_deriv_threshold : float
        Derivative threshold (dOD/dt) above which we say growth has 'started'.
    min_points_before_lag : int
        Number of extra points to keep before lag ends/growth has 'started' (default 0 = cut at threshold).
    post_peak_points : int
        How many points after the peak to keep (default 0 = cut at max).
    smooth_window : int
        Moving-average window for derivative smoothing. If 1, no smoothing is applied.
        
    Returns
    -------
    t_trim, od_trim : arrays
        Trimmed time and OD data.
    """

    time = np.asarray(time)
    od = np.asarray(od)
    assert len(time) == len(od)

    if trim_od_threshold > 0.0:
            mask = od >= trim_od_threshold
            time = time[mask]
            od = od[mask]
    
    if od.size == 0:
         return time, od
    
    # smooth OD slightly
    if smooth_window > 1:
        kernel = np.ones(smooth_window)/smooth_window
        od_smooth = np.convolve(od, kernel, mode='same')
    else:
        od_smooth = od.copy()
    
    # compute derivative dOD/dt
    dODdt = np.gradient(od_smooth, time)
    
    # --- find lag end ---
    # index where derivative first exceeds threshold and stays positive
    active = np.where(dODdt > lag_deriv_threshold)[0]
    if len(active) == 0:
        start_ix = 0
    else:
        start_ix = active[0]
        # back up a few points for safety
        start_ix = max(0, start_ix - min_points_before_lag)
    
    # --- find OD maximum ---
    peak_ix = int(np.argmax(od_smooth))
    end_ix = min(len(od)-1, peak_ix + post_peak_points)
    
    # --- trim data ---
    t_trim = time[start_ix:end_ix+1]
    od_trim = od[start_ix:end_ix+1]
    
    return t_trim, od_trim