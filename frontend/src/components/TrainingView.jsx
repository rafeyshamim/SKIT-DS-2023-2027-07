import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Database, 
  RefreshCw, 
  Award,
  Zap,
  BarChart3,
  Clock3
} from 'lucide-react';
import { fetchTrainingStatus, triggerTraining } from '../api';

export default function TrainingView() {
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [triggering, setTriggering] = useState(false);
  const [selectedEpochs, setSelectedEpochs] = useState(5);
  const [statusError, setStatusError] = useState('');

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  async function loadStatus() {
    setLoading(true);
    try {
      const data = await fetchTrainingStatus();
      setTrainingStatus(data);
      setStatusError('');
    } catch (err) {
      setTrainingStatus(null);
      setStatusError(err.message || 'Could not fetch training status.');
    } finally {
      setLoading(false);
    }
  }

  async function handleStartTraining() {
    setTriggering(true);
    try {
      const response = await triggerTraining(selectedEpochs);
      if (response.status !== 'STARTED') throw new Error(response.message || 'Training did not start.');
      await loadStatus();
    } catch (err) {
      setStatusError(err.message || 'Failed to start training.');
    } finally {
      setTriggering(false);
    }
  }

  const history = trainingStatus?.checkpoint?.history;
  const accuracyArr = history?.accuracy || [];
  const lossArr = history?.loss || [];
  const maxAcc = accuracyArr.length > 0 ? Math.max(...accuracyArr) : 0;
  const firstLoss = lossArr.length > 0 ? lossArr[0] : null;
  const lastLoss = lossArr.length > 0 ? lossArr[lossArr.length - 1] : null;
  const datasetLabel = trainingStatus?.dataset_name?.replace(/3d$/i, '3D').toUpperCase() || 'Dataset not configured';
  const architecture = trainingStatus?.architecture || {};
  const checkpoint = trainingStatus?.checkpoint;
  const evaluation = trainingStatus?.evaluation_metrics;
  const runHistory = [...(trainingStatus?.run_history || [])].reverse();
  const splits = trainingStatus?.dataset_splits;
  const formatMetric = (value, digits = 2) => value != null && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : 'Not recorded';
  const formatDate = (value) => value ? new Date(typeof value === 'number' ? value * 1000 : value).toLocaleString() : 'Not recorded';

  return (
    <div className="animate-fade-in">
      <div className="view-header">
        <div className="view-title-group">
          <h1>3D CNN Model Training & Online Dataset</h1>
          <p>Configured volumetric dataset, 3D CNN training, saved checkpoints, and recorded evaluation metrics.</p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button className="btn btn-secondary" onClick={loadStatus} disabled={loading}>
            <RefreshCw size={14} />
            <span>{loading ? 'Loading…' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {/* Top Telemetry Row */}
      {statusError && <p role="alert" style={{ color: '#f59e0b', marginBottom: '1rem' }}>{statusError}</p>}
      {trainingStatus?.status === 'ERROR' && <p role="alert" style={{ color: '#f59e0b', marginBottom: '1rem' }}>{trainingStatus.message}</p>}
      {!trainingStatus && !statusError && <p role="status" style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>Loading live training status…</p>}

      <div className="grid-3col" style={{ marginBottom: '1.75rem' }}>
        {/* Dataset Card */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.85rem' }}>
            <Database size={18} color="#06b6d4" />
            <h3 style={{ fontSize: '1.05rem', color: 'var(--text-main)' }}>Target Dataset</h3>
          </div>

          <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: '0.2rem' }}>
            {datasetLabel}
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: '0.85rem' }}>
            {trainingStatus?.dataset_source ? `Source: ${trainingStatus.dataset_source}` : 'Dataset source not configured'}
            {trainingStatus?.dataset_classes?.length ? ` · Classes: ${trainingStatus.dataset_classes.join(', ')}` : ''}
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', fontSize: '0.82rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Status:</span>
              <span style={{ color: 'var(--primary-hover)', fontWeight: 600 }}>
                {trainingStatus?.dataset_available ? `${trainingStatus.dataset_size_mb} MB · Available` : 'Dataset file not found'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Training Set:</span>
              <span className="font-mono" style={{ color: 'var(--text-main)' }}>{splits?.train ?? '—'} volumes</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Validation Set:</span>
              <span className="font-mono" style={{ color: 'var(--text-main)' }}>{splits?.val ?? '—'} volumes</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Test Set:</span>
              <span className="font-mono" style={{ color: 'var(--text-main)' }}>{splits?.test ?? '—'} volumes</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Volume shape:</span>
              <span className="font-mono" style={{ color: 'var(--text-main)' }}>{trainingStatus?.volume_shape?.join(' × ') || '—'}</span>
            </div>
          </div>
        </div>

        {/* Model Architecture Card */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.85rem' }}>
            <Cpu size={18} color="#06b6d4" />
            <h3 style={{ fontSize: '1.05rem', color: 'var(--text-main)' }}>3D CNN Architecture</h3>
          </div>

          <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: '0.2rem' }}>
            {architecture.input_shape ? `${architecture.input_shape.join(' × ')} input` : 'Architecture not configured'}
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: '0.85rem' }}>
            {architecture.optimizer ? `${architecture.optimizer} · ${architecture.loss || 'Loss not configured'}` : 'Model settings unavailable'}
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', fontSize: '0.82rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Conv3D Filters:</span>
              <span className="font-mono" style={{ color: '#0284c7' }}>{architecture.filters?.join(', ') || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Regularization:</span>
              <span style={{ color: 'var(--text-main)' }}>L2 = {architecture.l2_reg ?? '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Pooling:</span>
              <span style={{ color: 'var(--text-main)' }}>Kernel: {architecture.kernel_size?.join(' × ') || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Binary Head:</span>
              <span className="font-mono" style={{ color: 'var(--text-main)' }}>{architecture.dense_units?.join(' → ') || '—'}</span>
            </div>
          </div>
        </div>

        {/* Checkpoint Status Card */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.85rem' }}>
            <Award size={18} color="#06b6d4" />
            <h3 style={{ fontSize: '1.05rem', color: 'var(--text-main)' }}>Active Checkpoint</h3>
          </div>

          <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--primary-hover)', marginBottom: '0.2rem' }}>
            {checkpoint?.checkpoint_exists ? checkpoint.checkpoint_path.split(/[\\/]/).pop() : 'No checkpoint found'}
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: '0.85rem' }}>
            {checkpoint?.checkpoint_exists ? `${checkpoint.size_mb} MB · Updated ${formatDate(checkpoint.last_modified)}` : 'Train a model to create a checkpoint'}
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', fontSize: '0.82rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Training Accuracy:</span>
              <span className="font-mono" style={{ color: 'var(--primary-hover)', fontWeight: 700 }}>{formatMetric(accuracyArr.at(-1))}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Test Precision:</span>
              <span className="font-mono" style={{ color: 'var(--text-main)' }}>{formatMetric(evaluation?.precision)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Test AUC-ROC:</span>
              <span className="font-mono" style={{ color: '#0284c7' }}>{Number.isFinite(Number(evaluation?.auc_roc)) ? Number(evaluation.auc_roc).toFixed(4) : 'Not evaluated'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Model Status:</span>
              <span className="badge badge-low font-mono">{checkpoint?.checkpoint_exists ? 'AVAILABLE' : 'MISSING'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Live Training Trigger & Progress */}
      <div className="glass-panel" style={{ padding: '1.75rem', marginBottom: '1.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', color: 'var(--text-main)', marginBottom: '0.25rem' }}>
              Execute 3D CNN Training Cycle
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.84rem' }}>
            Runs the configured dataset through preprocessing, 3D CNN fitting, and checkpointing. Epoch metrics are saved after the run.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Epochs:</span>
              <select
                value={selectedEpochs}
                onChange={(e) => setSelectedEpochs(parseInt(e.target.value, 10))}
                className="form-select"
                style={{ padding: '0.35rem 0.65rem', fontSize: '0.82rem', width: 'auto' }}
              >
                <option value={3}>3 Epochs</option>
                <option value={5}>5 Epochs</option>
                <option value={10}>10 Epochs</option>
              </select>
            </div>

            <button
              id="btn-start-training"
              className="btn btn-primary"
              onClick={handleStartTraining}
              disabled={triggering || trainingStatus?.is_training}
            >
              <Zap size={15} />
              <span>{trainingStatus?.is_training ? 'Training Active...' : 'Start Training'}</span>
            </button>
          </div>
        </div>

        {trainingStatus?.is_training && (
          <div style={{ background: 'var(--primary-tint)', border: '1px solid var(--primary)', borderRadius: 'var(--radius-md)', padding: '0.85rem 1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem', color: 'var(--primary-hover)', fontWeight: 600, fontSize: '0.85rem' }}>
              <span>{trainingStatus.message}</span>
              <span className="badge badge-cyan font-mono">STATUS: {trainingStatus.status}</span>
            </div>
          </div>
        )}
      </div>

      {/* Epoch Metrics Progression Chart */}
      <div className="glass-panel" style={{ padding: '1.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem', marginBottom: '1.25rem' }}>
          <BarChart3 size={18} color="#06b6d4" />
          <h3 style={{ fontSize: '1.05rem', color: 'var(--text-main)' }}>Training Convergence History</h3>
        </div>

        <div className="grid-2col">
          {/* Accuracy Progression */}
          <div style={{ background: 'var(--bg-base)', padding: '1.25rem', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.85rem', display: 'flex', justifyContent: 'space-between' }}>
              <span>Accuracy per Epoch</span>
              <span className="font-mono" style={{ color: 'var(--primary-hover)', fontWeight: 700 }}>{accuracyArr.length ? `Peak: ${(maxAcc * 100).toFixed(1)}%` : 'No recorded history'}</span>
            </div>

            {accuracyArr.length ? <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.85rem', height: '130px', paddingTop: '0.85rem' }}>
              {accuracyArr.map((acc, idx) => {
                const heightPct = Math.round(acc * 100 * 1.1);
                return (
                  <div key={idx} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.35rem', height: '100%', justifyContent: 'flex-end' }}>
                    <span className="font-mono" style={{ fontSize: '0.72rem', color: 'var(--primary-hover)', fontWeight: 600 }}>
                      {(acc * 100).toFixed(1)}%
                    </span>
                    <div 
                      style={{ 
                        width: '100%', 
                        height: `${Math.max(12, heightPct)}%`, 
                        background: 'linear-gradient(to top, #06b6d4, #0891b2)', 
                        borderRadius: '4px 4px 0 0',
                        transition: 'height 0.4s'
                      }} 
                    />
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>E{idx + 1}</span>
                  </div>
                );
              })}
            </div> : <p style={{ color: 'var(--text-muted)' }}>No training accuracy has been recorded yet.</p>}
          </div>

          {/* Loss Reduction */}
          <div style={{ background: 'var(--bg-base)', padding: '1.25rem', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.85rem', display: 'flex', justifyContent: 'space-between' }}>
              <span>Loss per Epoch (Convergence)</span>
              <span className="font-mono" style={{ color: '#0284c7' }}>{firstLoss == null ? 'Not recorded' : `${firstLoss.toFixed(3)} → ${lastLoss.toFixed(3)}`}</span>
            </div>

            {lossArr.length ? <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.85rem', height: '130px', paddingTop: '0.85rem' }}>
              {lossArr.map((l, idx) => {
                const heightPct = Math.round((l / 2.8) * 100);
                return (
                  <div key={idx} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.35rem', height: '100%', justifyContent: 'flex-end' }}>
                    <span className="font-mono" style={{ fontSize: '0.72rem', color: '#0284c7', fontWeight: 600 }}>
                      {l.toFixed(2)}
                    </span>
                    <div 
                      style={{ 
                        width: '100%', 
                        height: `${Math.max(12, heightPct)}%`, 
                        background: 'linear-gradient(to top, #0284c7, #38bdf8)', 
                        borderRadius: '4px 4px 0 0',
                        transition: 'height 0.4s'
                      }} 
                    />
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>E{idx + 1}</span>
                  </div>
                );
              })}
            </div> : <p style={{ color: 'var(--text-muted)' }}>No training loss has been recorded yet.</p>}
          </div>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '1.75rem', marginTop: '1.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem', marginBottom: '1.25rem' }}>
          <Clock3 size={18} color="#06b6d4" />
          <h3 style={{ fontSize: '1.05rem', color: 'var(--text-main)' }}>Saved Training Runs</h3>
        </div>
        {runHistory.length ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead><tr>
                <th style={{ padding: '0.65rem' }}>Recorded</th>
                <th style={{ padding: '0.65rem' }}>Dataset</th>
                <th style={{ padding: '0.65rem' }}>Status</th>
                <th style={{ padding: '0.65rem' }}>Epochs</th>
                <th style={{ padding: '0.65rem' }}>Final train accuracy</th>
                <th style={{ padding: '0.65rem' }}>Final validation accuracy</th>
              </tr></thead>
              <tbody>{runHistory.map((run) => (
                <tr key={run.run_id}>
                  <td style={{ padding: '0.65rem' }}>{formatDate(run.recorded_at || run.finished_at || run.started_at)}</td>
                  <td style={{ padding: '0.65rem' }}>{run.dataset_name || '—'}</td>
                  <td style={{ padding: '0.65rem' }}>{run.status || '—'}</td>
                  <td style={{ padding: '0.65rem' }}>{run.completed_epochs ?? '—'} / {run.requested_epochs ?? '—'}</td>
                  <td style={{ padding: '0.65rem' }}>{formatMetric(run.metrics?.accuracy?.at(-1))}</td>
                  <td style={{ padding: '0.65rem' }}>{formatMetric(run.metrics?.val_accuracy?.at(-1))}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>No training runs have been recorded. Completed or failed runs will be saved here.</p>
        )}
        <p style={{ color: 'var(--text-muted)', fontSize: '0.78rem', marginTop: '0.8rem' }}>
          Run history is persisted in results/training_runs.json; the latest epoch-by-epoch metrics are in results/training_history.json.
        </p>
      </div>
    </div>
  );
}
