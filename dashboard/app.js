"use strict";

const NS = "http://www.w3.org/2000/svg";
const state = { data: null, interval: "3m" };
const $ = (selector, root = document) => root.querySelector(selector);

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function svg(tag, attrs = {}, text) {
  const el = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
  if (text !== undefined) el.textContent = text;
  return el;
}

function text(parent, attrs, value) {
  parent.appendChild(svg("text", attrs, value));
}

function scale(d0, d1, r0, r1) {
  return (value) => r0 + ((value - d0) / (d1 - d0 || 1)) * (r1 - r0);
}

function extent(values, pad = 0.12) {
  let min = Math.min(0, ...values);
  let max = Math.max(0, ...values);
  const span = Math.max(max - min, 1);
  return [min - span * pad, max + span * pad];
}

function ticks(min, max, count = 5) {
  const rough = Math.max((max - min) / count, 0.1);
  const power = 10 ** Math.floor(Math.log10(rough));
  const ratio = rough / power;
  const step = (ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10) * power;
  const values = [];
  for (let value = Math.ceil(min / step) * step; value <= max + step * 0.01; value += step) {
    values.push(Number(value.toFixed(10)));
  }
  return values;
}

function signed(value) {
  if (Math.abs(value) < 0.0005) return "0.000";
  return `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
}

function path(values, x, y) {
  return values.map((value, index) => `${index ? "L" : "M"}${x(index).toFixed(2)},${y(value).toFixed(2)}`).join(" ");
}

function title(svgNode, heading, description) {
  const id = Math.random().toString(36).slice(2);
  svgNode.setAttribute("role", "img");
  svgNode.setAttribute("aria-labelledby", `title-${id} desc-${id}`);
  svgNode.append(
    svg("title", { id: `title-${id}` }, heading),
    svg("desc", { id: `desc-${id}` }, description)
  );
}

function axes(svgNode, config) {
  const { width, height, left, right, top, bottom, yMin, yMax, y, x, xValues, xLabel, yLabel } = config;
  ticks(yMin, yMax).forEach((value) => {
    const yPos = y(value);
    svgNode.appendChild(svg("line", {
      x1: left,
      x2: width - right,
      y1: yPos,
      y2: yPos,
      class: Math.abs(value) < 1e-8 ? "zero-line" : "gridline"
    }));
    text(svgNode, { x: left - 12, y: yPos + 4, "text-anchor": "end" }, Number(value.toFixed(1)).toString());
  });
  xValues.forEach((value, index) => {
    const xPos = x(index);
    svgNode.appendChild(svg("line", { x1: xPos, x2: xPos, y1: top, y2: height - bottom, class: "gridline" }));
    text(svgNode, { x: xPos, y: height - bottom + 26, "text-anchor": "middle" }, xLabel(value));
  });
  text(svgNode, { x: left, y: 15, class: "axis-label" }, yLabel);
}

let chartObserver;

function reveal(mount) {
  mount.classList.remove("is-visible");
  if (matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) {
    mount.classList.add("is-visible");
    return;
  }
  chartObserver ||= new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        chartObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.22 });
  requestAnimationFrame(() => chartObserver.observe(mount));
}

function point(parent, attrs, tooltip, order) {
  const circle = svg("circle", { ...attrs, class: `${attrs.class} chart-point` });
  circle.style.setProperty("--point-order", order);
  circle.appendChild(svg("title", {}, tooltip));
  parent.appendChild(circle);
}

function signalPanel(dataset, series, yMin, yMax) {
  const width = 420;
  const height = 330;
  const left = 52;
  const right = 18;
  const top = 28;
  const bottom = 52;
  const x = scale(0, dataset.rhos.length - 1, left, width - right);
  const y = scale(yMin, yMax, height - bottom, top);
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "chart-svg", preserveAspectRatio: "xMidYMid meet" });

  title(chart, `${series.label} signal quality response`, "Thick line is the mean. Thin lines and points are individual training seeds.");
  axes(chart, {
    width, height, left, right, top, bottom, yMin, yMax, y, x,
    xValues: dataset.rhos,
    xLabel: (value) => `rho=${value}`,
    yLabel: "signal increment (bps)"
  });

  series.seeds.forEach((seed, seedIndex) => {
    chart.appendChild(svg("path", {
      d: path(seed, x, y),
      pathLength: 1,
      class: "seed-line animated-path"
    }));
    seed.forEach((value, index) => {
      point(chart, { cx: x(index), cy: y(value), r: 3, class: "seed-point" }, `seed ${seedIndex}: ${signed(value)} bps`, index);
    });
  });

  chart.appendChild(svg("path", {
    d: path(series.mean, x, y),
    pathLength: 1,
    class: "mean-line animated-path"
  }));
  series.mean.forEach((value, index) => {
    point(chart, { cx: x(index), cy: y(value), r: 5, class: "mean-point" }, `mean: ${signed(value)} bps`, index);
  });
  const last = series.mean.length - 1;
  const panel = node("article", "signal-panel");
  const header = node("div", "signal-panel-head");
  header.append(
    node("h3", "", series.label),
    node("span", "signal-panel-value", `rho=0.8 ${signed(series.mean[last])}`)
  );
  panel.append(header);
  panel.append(chart);
  return panel;
}

function drawSignal(dataset) {
  const mount = $("#signal-chart");
  const values = dataset.series.flatMap((series) => [...series.mean, ...series.seeds.flat()]);
  const [yMin, yMax] = extent(values);
  mount.replaceChildren(
    ...dataset.series.map((series) => signalPanel(dataset, series, yMin, yMax))
  );
  reveal(mount);
}

function drawTwap(dataset) {
  const mount = $("#twap-chart");
  const width = 1000;
  const height = 400;
  const left = 126;
  const right = 46;
  const top = 38;
  const bottom = 54;
  const values = dataset.methods.flatMap((method) => [method.mean, ...method.seeds]);
  const [xMin, xMax] = extent(values, 0.08);
  const x = scale(xMin, xMax, left, width - right);
  const row = (height - top - bottom) / dataset.methods.length;
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "chart-svg", preserveAspectRatio: "xMidYMid meet" });

  title(chart, "Improvement versus TWAP", "Diamonds are method means. Circles are individual training seeds, connected by their minimum-to-maximum range. Positive values indicate lower implementation shortfall than TWAP.");
  ticks(xMin, xMax).forEach((value) => {
    const xPos = x(value);
    chart.appendChild(svg("line", {
      x1: xPos,
      x2: xPos,
      y1: top - 8,
      y2: height - bottom,
      class: Math.abs(value) < 1e-8 ? "zero-line" : "gridline"
    }));
    text(chart, { x: xPos, y: height - 17, "text-anchor": "middle" }, Number(value.toFixed(1)).toString());
  });

  dataset.methods.forEach((method, index) => {
    const y = top + row * index + row / 2;
    const mean = x(method.mean);
    const minSeed = Math.min(...method.seeds);
    const maxSeed = Math.max(...method.seeds);
    const hasMultipleSeeds = method.seeds.length > 1;
    text(chart, { x: left - 16, y: y + 4, "text-anchor": "end", class: "axis-label" }, method.label);
    if (hasMultipleSeeds) {
      chart.appendChild(svg("line", { x1: x(minSeed), x2: x(maxSeed), y1: y, y2: y, pathLength: 1, class: "range-line animated-path" }));
      method.seeds.forEach((value, seedIndex) => {
        point(chart, { cx: x(value), cy: y, r: 5.5, class: "twap-seed" }, `${method.label} seed ${seedIndex}: ${signed(value)} bps`, seedIndex);
      });
    }
    const diamond = svg("polygon", {
      points: `${mean},${y - 8} ${mean + 8},${y} ${mean},${y + 8} ${mean - 8},${y}`,
      class: "mean-diamond chart-point"
    });
    diamond.style.setProperty("--point-order", 3);
    diamond.appendChild(svg("title", {}, `${method.label}${hasMultipleSeeds ? " mean" : ""}: ${signed(method.mean)} bps`));
    chart.appendChild(diamond);
    text(chart, { x: mean, y: y - 15, "text-anchor": "middle", class: "axis-label" }, signed(method.mean));
  });
  text(chart, { x: width - right, y: 16, "text-anchor": "end", class: "axis-label" }, "Improvement vs TWAP (bps)  ·  ← worse   better →");

  mount.replaceChildren(chart);
  reveal(mount);
}

function empty(mount, interval, kind) {
  mount.classList.remove("is-visible");
  const wrap = node("div", "empty-state");
  const copy = node("div");
  const heading = interval.status === "in-progress"
    ? "Experiment running"
    : "Results pending";
  copy.append(node("strong", "", heading));
  wrap.appendChild(copy);
  mount.replaceChildren(wrap);
}

function renderInterval() {
  const interval = state.data.intervals.find((item) => item.id === state.interval);
  $("#conclusion").textContent = interval.conclusion || "";

  if (!interval.chartData) {
    $("#signal-tools").hidden = true;
    $("#twap-legend").hidden = true;
    empty($("#signal-chart"), interval, "signal");
    empty($("#twap-chart"), interval, "twap");
    $("#signal-scope").textContent = `${interval.label} · results pending`;
    return;
  }

  $("#signal-tools").hidden = false;
  $("#twap-legend").hidden = false;
  $("#signal-scope").textContent = interval.chartData.signalResponse.scope;
  drawSignal(interval.chartData.signalResponse);
  drawTwap(interval.chartData.vsTwap);
}

function renderControls() {
  const root = $("#interval-control");
  state.data.intervals.forEach((interval) => {
    const wrap = node("div", "interval-option");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "interval";
    input.value = interval.id;
    input.id = `interval-${interval.id}`;
    input.checked = interval.id === state.interval;
    const label = node("label", "", interval.label);
    label.htmlFor = input.id;
    wrap.append(input, label);
    root.appendChild(wrap);
  });
  root.addEventListener("change", (event) => {
    if (event.target.name === "interval") {
      state.interval = event.target.value;
      renderInterval();
    }
  });
}

async function init() {
  try {
    const response = await fetch("data/experiments.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    state.interval = state.data.meta.defaultInterval;
    renderControls();
    renderInterval();
  } catch (error) {
    $("#signal-scope").textContent = `Data load error · ${error.message}`;
    empty($("#signal-chart"), { label: "Data", status: "data-ready" }, "signal");
    empty($("#twap-chart"), { label: "Data", status: "data-ready" }, "twap");
  }
}

init();
