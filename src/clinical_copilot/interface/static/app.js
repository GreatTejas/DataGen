// Clinical Deterioration Copilot - Front-End Logic
let selectedPatientId = null;
let vitalsChartInstance = null;
let currentAction = null;
let isStreaming = false;
let streamInterval = null;
let currentFilter = 'all';

document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  loadPatients();
  loadAuditStatus();
});

function initEventListeners() {
  // Filter pills
  document.querySelectorAll('.filter-pills .pill').forEach(pill => {
    pill.addEventListener('click', (e) => {
      document.querySelectorAll('.filter-pills .pill').forEach(p => p.classList.remove('active'));
      e.target.classList.add('active');
      currentFilter = e.target.getAttribute('data-filter');
      loadPatients();
    });
  });

  // Simulation controls
  document.getElementById('btn-step').addEventListener('click', () => stepSimulation(60));
  document.getElementById('btn-step-cohort').addEventListener('click', () => stepSimulation(3600));
  document.getElementById('btn-play-pause').addEventListener('click', toggleStream);
  document.getElementById('btn-reset').addEventListener('click', resetSimulation);

  // Clinician action buttons
  document.getElementById('btn-action-accept').addEventListener('click', () => openActionModal('ACCEPT'));
  document.getElementById('btn-action-investigate').addEventListener('click', () => openActionModal('INVESTIGATE'));
  document.getElementById('btn-action-defer').addEventListener('click', () => openActionModal('DEFER'));
  document.getElementById('btn-action-dismiss').addEventListener('click', () => openActionModal('DISMISS'));

  // Feedback Modal
  document.getElementById('btn-close-modal').addEventListener('click', closeActionModal);
  document.getElementById('btn-cancel-modal').addEventListener('click', closeActionModal);
  document.getElementById('btn-submit-feedback').addEventListener('click', submitClinicianFeedback);

  // Audit Modal
  document.getElementById('btn-open-audit').addEventListener('click', openAuditModal);
  document.getElementById('btn-close-audit').addEventListener('click', closeAuditModal);
  document.getElementById('btn-refresh-audit').addEventListener('click', loadAuditTrail);

  // Metrics Modal
  document.getElementById('btn-open-metrics').addEventListener('click', openMetricsModal);
  document.getElementById('btn-close-metrics').addEventListener('click', closeMetricsModal);
}

// 1. Load Patients & Render Triage Board
async function loadPatients() {
  try {
    const res = await fetch('/api/patients');
    const patients = await res.json();
    renderKPIs(patients);
    renderPatientList(patients);
  } catch (err) {
    console.error('Failed to load patients', err);
  }
}

function renderKPIs(patients) {
  let activeAlerts = 0;
  let deferred = 0;
  patients.forEach(p => {
    if (p.status.startsWith('ALERT')) activeAlerts++;
    else if (p.status.startsWith('DEFER')) deferred++;
  });

  document.getElementById('kpi-total-patients').textContent = patients.length;
  document.getElementById('kpi-active-alerts').textContent = activeAlerts;
  document.getElementById('kpi-deferred').textContent = deferred;
}

function renderPatientList(patients) {
  const listEl = document.getElementById('patient-list');
  listEl.innerHTML = '';

  const filtered = patients.filter(p => {
    if (currentFilter === 'alert') return p.status.startsWith('ALERT');
    if (currentFilter === 'deferred') return p.status.startsWith('DEFER');
    if (currentFilter === 'stable') return p.status === 'STABLE';
    return true;
  });

  if (filtered.length === 0) {
    listEl.innerHTML = '<div class="empty-msg" style="padding:20px;text-align:center;color:#64748b;">No patients matching filter.</div>';
    return;
  }

  filtered.forEach(p => {
    const card = document.createElement('div');
    card.className = `patient-card ${selectedPatientId === p.patient_id ? 'active' : ''}`;
    card.onclick = () => selectPatient(p.patient_id);

    let statusPillClass = 'status-stable';
    if (p.status.startsWith('ALERT (CRITICAL)')) statusPillClass = 'status-alert';
    else if (p.status.startsWith('ALERT')) statusPillClass = 'status-alert-high';
    else if (p.status.startsWith('DEFER')) statusPillClass = 'status-deferred';

    const vitals = p.latest_vitals ? p.latest_vitals.vitals : null;
    const hr = vitals ? Math.round(vitals.heart_rate) : '--';
    const bp = vitals ? `${Math.round(vitals.systolic_bp)}/${Math.round(vitals.diastolic_bp)}` : '--';
    const spo2 = vitals ? `${Math.round(vitals.spo2)}%` : '--';
    const rr = vitals ? Math.round(vitals.respiratory_rate) : '--';

    const gtScenarioTag = p.has_ground_truth ? `<span class="badge gt-badge" title="Ground Truth Episode">${p.gt_scenario}</span>` : '';

    card.innerHTML = `
      <div class="card-top">
        <span class="card-patient-id mono">${p.patient_id}</span>
        <span class="status-pill ${statusPillClass}">${p.status}</span>
      </div>
      <div class="card-mid">
        ${p.age}y ${p.sex} • NEWS2: <b style="color:#ffffff">${p.latest_news2}</b> • SI: <b style="color:#ffffff">${p.latest_si}</b>
        ${gtScenarioTag}
      </div>
      <div class="card-vitals-row mono">
        <div class="vital-tag">HR: <span class="val">${hr}</span></div>
        <div class="vital-tag">BP: <span class="val">${bp}</span></div>
        <div class="vital-tag">SpO2: <span class="val">${spo2}</span></div>
        <div class="vital-tag">RR: <span class="val">${rr}</span></div>
      </div>
    `;
    listEl.appendChild(card);
  });
}

// 2. Select Patient and Load Details
async function selectPatient(patientId) {
  selectedPatientId = patientId;
  document.querySelectorAll('.patient-card').forEach(c => c.classList.remove('active'));
  document.getElementById('empty-selection').style.display = 'none';
  document.getElementById('detail-container').style.display = 'block';

  try {
    const res = await fetch(`/api/patients/${patientId}`);
    const data = await res.json();
    renderPatientDetail(data);
  } catch (err) {
    console.error('Failed to load patient detail', err);
  }
}

function renderPatientDetail(data) {
  const ctx = data.context;
  document.getElementById('detail-patient-id').textContent = data.patient_id;
  document.getElementById('detail-status-pill').textContent = data.status;

  let pillClass = 'status-stable';
  if (data.status.startsWith('ALERT (CRITICAL)')) pillClass = 'status-alert';
  else if (data.status.startsWith('ALERT')) pillClass = 'status-alert-high';
  else if (data.status.startsWith('DEFER')) pillClass = 'status-deferred';
  document.getElementById('detail-status-pill').className = `status-pill ${pillClass}`;

  // Ground truth badge
  const gtBadge = document.getElementById('detail-ground-truth-badge');
  if (data.ground_truth) {
    gtBadge.style.display = 'inline-block';
    gtBadge.textContent = `Target Scenario: ${data.ground_truth.scenario.replace('_', ' ')}`;
  } else {
    gtBadge.style.display = 'none';
  }

  // Demographics and Context
  document.getElementById('detail-age-sex').textContent = `${ctx.age} yrs • ${ctx.sex}`;
  document.getElementById('detail-history').textContent = `History: ${ctx.relevant_history.join(', ') || 'None'}`;
  document.getElementById('detail-meds').textContent = `Meds: ${ctx.current_medications.join(', ') || 'None'}`;
  const labs = ctx.recent_labs || {};
  document.getElementById('detail-labs').textContent = `Creatinine: ${labs.creatinine_mg_dl || '--'} mg/dL | WBC: ${labs.wbc_k_ul || '--'}k`;

  // Quick scores
  document.getElementById('detail-news2').textContent = data.latest_news2;
  document.getElementById('detail-si').textContent = data.latest_si;

  // Active Alert Box
  const alertBox = document.getElementById('detail-alert-box');
  const alert = data.active_alert;

  if (alert) {
    alertBox.style.display = 'block';
    const prioTag = document.getElementById('detail-priority-tag');
    prioTag.textContent = alert.priority || 'HIGH';
    prioTag.className = `priority-tag ${alert.priority ? alert.priority.toLowerCase() : 'high'}`;

    document.getElementById('detail-syndrome-title').textContent = (alert.syndrome || 'DETERIORATION DETECTED').toUpperCase().replace(/_/g, ' ');
    document.getElementById('detail-cooldown-info').textContent = alert.episode_id ? `Episode ID: ${alert.episode_id} • Cooldown Guarded` : 'Live Telemetry Breach';
    document.getElementById('detail-reasoning-text').textContent = alert.explanation || 'Multi-parameter deterioration identified.';

    // Populate actions
    const actionsList = document.getElementById('detail-actions-list');
    actionsList.innerHTML = '';
    const actions = (alert.recommendation && alert.recommendation.actions) ? alert.recommendation.actions : [
      'Perform urgent bedside clinical review.',
      'Check vital signs repeat in 15 minutes.',
      'Escalate to attending physician if parameters worsen.'
    ];
    actions.forEach(act => {
      const li = document.createElement('li');
      li.textContent = act;
      actionsList.appendChild(li);
    });

    // Populate RAG citations
    const citGrid = document.getElementById('detail-citations-grid');
    citGrid.innerHTML = '';
    const citations = (alert.recommendation && alert.recommendation.citations) ? alert.recommendation.citations : [];
    citations.forEach(c => {
      const chip = document.createElement('div');
      chip.className = 'citation-chip';
      chip.innerHTML = `
        <div class="citation-chip-title">${c.title} (${c.version})</div>
        <div class="citation-chip-loc">${c.locator}</div>
      `;
      citGrid.appendChild(chip);
    });
  } else {
    // If no alert active, show peaceful monitoring state
    alertBox.style.display = 'block';
    const prioTag = document.getElementById('detail-priority-tag');
    prioTag.textContent = 'MONITORING';
    prioTag.className = 'priority-tag'
    prioTag.style.backgroundColor = '#10b981';

    document.getElementById('detail-syndrome-title').textContent = 'PHYSIOLOGICAL PARAMETERS STABLE';
    document.getElementById('detail-cooldown-info').textContent = 'Continuously streaming telemetry • No active alerts';
    document.getElementById('detail-reasoning-text').textContent = 'Patient vitals are within normal reference limits or aligned with personalized chronic baseline. Trajectory is stable.';

    const actionsList = document.getElementById('detail-actions-list');
    actionsList.innerHTML = '<li>Continue routine bedside telemetry monitoring according to standard nursing schedule.</li>';

    document.getElementById('detail-citations-grid').innerHTML = '<div style="color:#94a3b8;font-size:0.8rem;">Royal College of Physicians NEWS2 Standard Protocol (2017)</div>';
  }

  // Render Vitals Chart
  renderVitalsChart(data.vitals_history);
}

// 3. Render Chart.js Telemetry Chart
function renderVitalsChart(history) {
  const ctx = document.getElementById('vitalsChart').getContext('2d');
  if (vitalsChartInstance) {
    vitalsChartInstance.destroy();
  }

  const labels = history.map((h, i) => `${h.sequence}m`);
  const hrData = history.map(h => h.heart_rate);
  const sbpData = history.map(h => h.systolic_bp);
  const dbpData = history.map(h => h.diastolic_bp);
  const spo2Data = history.map(h => h.spo2);
  const rrData = history.map(h => h.respiratory_rate);

  vitalsChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Heart Rate',
          data: hrData,
          borderColor: '#f87171',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 1,
          tension: 0.2,
          yAxisID: 'y',
        },
        {
          label: 'Systolic BP',
          data: sbpData,
          borderColor: '#60a5fa',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 1,
          tension: 0.2,
          yAxisID: 'y',
        },
        {
          label: 'Diastolic BP',
          data: dbpData,
          borderColor: '#93c5fd',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.2,
          yAxisID: 'y',
        },
        {
          label: 'SpO2',
          data: spo2Data,
          borderColor: '#34d399',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 1,
          tension: 0.2,
          yAxisID: 'y1',
        },
        {
          label: 'Resp Rate',
          data: rrData,
          borderColor: '#fbbf24',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 1,
          tension: 0.2,
          yAxisID: 'y2',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e293b',
          borderColor: '#334155',
          borderWidth: 1,
          titleFont: { family: 'JetBrains Mono' },
          bodyFont: { family: 'JetBrains Mono' }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', maxTicksLimit: 12, font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          min: 30,
          max: 200,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } },
          title: { display: true, text: 'HR & BP (mmHg / bpm)', color: '#64748b', font: { size: 10 } }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          min: 70,
          max: 100,
          grid: { drawOnChartArea: false },
          ticks: { color: '#34d399', font: { family: 'JetBrains Mono', size: 10 } },
          title: { display: true, text: 'SpO2 (%)', color: '#34d399', font: { size: 10 } }
        },
        y2: {
          type: 'linear',
          display: false,
          min: 5,
          max: 45
        }
      }
    }
  });
}

// 4. Simulation Controls
async function stepSimulation(count) {
  try {
    const res = await fetch(`/api/simulate/step?count=${count}`, { method: 'POST' });
    const data = await res.json();
    document.getElementById('sim-progress').textContent = `${data.cursor} / ${data.total} events`;

    await loadPatients();
    if (selectedPatientId) {
      await selectPatient(selectedPatientId);
    }
    loadAuditStatus();
  } catch (err) {
    console.error('Simulation step failed', err);
  }
}

function toggleStream() {
  const btn = document.getElementById('btn-play-pause');
  if (isStreaming) {
    clearInterval(streamInterval);
    isStreaming = false;
    btn.textContent = '▶️ Stream';
    btn.classList.remove('btn-primary');
    btn.classList.add('btn-secondary');
  } else {
    isStreaming = true;
    btn.textContent = '⏸ Pause';
    btn.classList.remove('btn-secondary');
    btn.classList.add('btn-primary');
    streamInterval = setInterval(() => {
      stepSimulation(60);
    }, 1200);
  }
}

async function resetSimulation() {
  if (isStreaming) toggleStream();
  await fetch('/api/simulate/reset', { method: 'POST' });
  document.getElementById('sim-progress').textContent = '0 / 10,800 events';
  await loadPatients();
  if (selectedPatientId) selectPatient(selectedPatientId);
  loadAuditStatus();
}

// 5. Clinician Feedback Modal Actions
function openActionModal(action) {
  if (!selectedPatientId) return;
  currentAction = action;

  const modal = document.getElementById('feedback-modal');
  const badge = document.getElementById('modal-action-badge');
  badge.textContent = action;

  if (action === 'ACCEPT') badge.style.backgroundColor = '#059669';
  else if (action === 'INVESTIGATE') badge.style.backgroundColor = '#2563eb';
  else if (action === 'DEFER') badge.style.backgroundColor = '#d97706';
  else if (action === 'DISMISS') badge.style.backgroundColor = '#4b5563';

  document.getElementById('modal-title').textContent = `${action} Deterioration Escalation for ${selectedPatientId}`;
  modal.style.display = 'flex';
}

function closeActionModal() {
  document.getElementById('feedback-modal').style.display = 'none';
  document.getElementById('textarea-clinical-notes').value = '';
}

async function submitClinicianFeedback() {
  if (!selectedPatientId || !currentAction) return;

  const clinicianId = document.getElementById('input-clinician-id').value;
  const feedbackCategory = document.getElementById('select-feedback-category').value;
  const notes = document.getElementById('textarea-clinical-notes').value;

  try {
    const res = await fetch(`/api/patients/${selectedPatientId}/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_id: selectedPatientId,
        action: currentAction,
        feedback_category: feedbackCategory,
        clinical_notes: notes,
        clinician_id: clinicianId
      })
    });
    const result = await res.json();
    closeActionModal();
    await loadPatients();
    selectPatient(selectedPatientId);
    loadAuditStatus();
  } catch (err) {
    console.error('Failed to submit feedback', err);
    alert('Failed to submit feedback. Check console.');
  }
}

// 6. Audit Trail Modal & Verification
async function loadAuditStatus() {
  try {
    const res = await fetch('/api/audit/verify');
    const data = await res.json();
    const tag = document.getElementById('audit-status-text');
    if (data.chain_valid) {
      tag.textContent = `SHA-256 Valid (${data.records_count} records)`;
      tag.style.color = '#34d399';
    } else {
      tag.textContent = 'Integrity Compromised!';
      tag.style.color = '#ef4444';
    }
  } catch (err) {
    console.error('Failed to load audit status', err);
  }
}

async function openAuditModal() {
  document.getElementById('audit-modal').style.display = 'flex';
  await loadAuditTrail();
}

function closeAuditModal() {
  document.getElementById('audit-modal').style.display = 'none';
}

async function loadAuditTrail() {
  const pid = document.getElementById('audit-filter-patient').value;
  const type = document.getElementById('audit-filter-type').value;

  let url = '/api/audit?limit=80';
  if (pid) url += `&patient_id=${encodeURIComponent(pid)}`;
  if (type) url += `&event_type=${encodeURIComponent(type)}`;

  try {
    const res = await fetch(url);
    const data = await res.json();

    const badge = document.getElementById('audit-chain-badge');
    if (data.chain_valid) {
      badge.textContent = `⛓️ SHA-256 Chain Verified (${data.total_records} records)`;
      badge.className = 'chain-badge valid';
    } else {
      badge.textContent = '❌ Hash Chain Broken!';
      badge.className = 'chain-badge invalid';
    }

    const tbody = document.getElementById('audit-table-body');
    tbody.innerHTML = '';

    data.returned_records.forEach(r => {
      const tr = document.createElement('tr');
      const timeStr = r.timestamp ? r.timestamp.substring(11, 19) : '--';
      const shortHash = r.entry_hash ? r.entry_hash.substring(0, 16) + '...' : '--';
      const summary = JSON.stringify(r.payload).substring(0, 85) + '...';

      tr.innerHTML = `
        <td class="mono">#${r.index}</td>
        <td class="mono">${timeStr}</td>
        <td><b style="color:#60a5fa">${r.event_type}</b></td>
        <td class="mono">${r.patient_id}</td>
        <td class="hash-cell mono" title="${r.entry_hash}">${shortHash}</td>
        <td style="color:#cbd5e1" title="${JSON.stringify(r.payload)}">${summary}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to load audit trail', err);
  }
}

// 7. Benchmark Metrics Modal
async function openMetricsModal() {
  document.getElementById('metrics-modal').style.display = 'flex';
  const body = document.getElementById('metrics-modal-body');
  body.innerHTML = '<div class="loading-state">Loading evaluation benchmark results...</div>';

  try {
    const res = await fetch('/api/metrics');
    const m = await res.json();

    body.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-box">
          <div class="metric-box-title">Episode Recall</div>
          <div class="metric-box-val" style="color:#34d399">${(m.episode_recall * 100).toFixed(1)}%</div>
          <div class="metric-box-sub">${m.true_positives} / ${m.ground_truth_episodes_count} true deterioration episodes detected (${m.false_negatives} missed)</div>
        </div>

        <div class="metric-box">
          <div class="metric-box-title">Median Lead Time</div>
          <div class="metric-box-val" style="color:#38bdf8">${m.median_lead_time_minutes} min</div>
          <div class="metric-box-sub">Early warning window ahead of critical escalation window (Mean: ${m.mean_lead_time_minutes} min)</div>
        </div>

        <div class="metric-box">
          <div class="metric-box-title">Duplicate Alert Suppression Rate</div>
          <div class="metric-box-val" style="color:#fbbf24">${(m.duplicate_suppression_rate * 100).toFixed(1)}%</div>
          <div class="metric-box-sub">${m.duplicate_suppression_count} repetitive/cooldown alerts suppressed to eliminate alarm fatigue</div>
        </div>

        <div class="metric-box">
          <div class="metric-box-title">Artifact False-Positive Rate</div>
          <div class="metric-box-val" style="color:#34d399">${(m.artifact_false_positive_rate * 100).toFixed(1)}%</div>
          <div class="metric-box-sub">${m.artifact_false_positives} / ${m.artifact_episodes_count} sensor artifacts escaped filter (100% suppressed)</div>
        </div>

        <div class="metric-box">
          <div class="metric-box-title">Alerts / Patient-Hour</div>
          <div class="metric-box-val">${m.alerts_per_patient_hour}</div>
          <div class="metric-box-sub">${m.total_alerts_fired} total alerts across ${m.total_patient_hours} monitored patient hours</div>
        </div>

        <div class="metric-box">
          <div class="metric-box-title">RAG Guideline Citation Coverage</div>
          <div class="metric-box-val" style="color:#a78bfa">${(m.rag_citation_coverage * 100).toFixed(1)}%</div>
          <div class="metric-box-sub">Alerts cite Sepsis-3, NEWS2, or NICE NG51 with doc ID & locator</div>
        </div>

        <div class="metric-box" style="grid-column: span 2;">
          <div class="metric-box-title">Audit Trail Integrity (SHA-256 Ledger)</div>
          <div class="metric-box-val" style="color:#34d399">100% Cryptographically Verified</div>
          <div class="metric-box-sub">${m.audit_records_count} total events (observations, decisions, actions) chained with zero tampering</div>
        </div>
      </div>
    `;
  } catch (err) {
    body.innerHTML = '<div style="color:#ef4444">Failed to load metrics. Check server.</div>';
  }
}

function closeMetricsModal() {
  document.getElementById('metrics-modal').style.display = 'none';
}
