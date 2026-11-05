package org.minorproject;

import ch.qos.logback.classic.Level;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerSpaceShared;
import org.cloudsimplus.util.Log;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * ML-Based Predictive Load Balancing and Auto-Scaling in Cloud Environments
 *
 * Replays the NASA HTTP workload (test period from predictions.csv) in CloudSim Plus
 * under four auto-scaling policies and compares SLA and cost:
 *
 *   1. STATIC          - fixed number of VMs sized for peak load (over-provisioning baseline)
 *   2. REACTIVE        - threshold scaler: looks at utilisation of the previous interval,
 *                        scales up above 80 %, scales down below 40 %
 *   3. PREDICTIVE_MA   - proactive scaler driven by a moving-average forecast
 *   4. PREDICTIVE_LSTM - proactive scaler driven by the LSTM forecast
 *
 * Modelling:
 *   - Each 5-minute step's requests arrive spread evenly across the step.
 *   - Requests are grouped into cloudlets of BATCH requests; one VM can serve
 *     VM_CAPACITY requests per step at 100 % load.
 *   - Load balancing: each request batch goes round-robin to the VMs that are
 *     active AND already booted at its arrival time.
 *   - Newly added VMs need BOOT_TIME seconds before they can serve (cold start).
 *   - Cost = active VM-hours x PRICE_PER_VM_HOUR.
 *   - SLA violation = request batch whose response time exceeds SLA_RESPONSE_TIME.
 */
public class AutoScalingSimulation {

    // parameters
    static final String CSV_PATH = "predictions.csv";
    static final String OUT_DIR = "results";

    static final double STEP = 300;                 // seconds per interval (5 min)
    static final int VM_CAPACITY = 100;             // requests one VM serves per step at full load
    static final double VM_MIPS = 1000;
    static final int BATCH = 5;                     // requests per cloudlet
    static final double BOOT_TIME = 120;            // VM cold-start delay (s)
    static final double SLA_RESPONSE_TIME = 60;     // response-time limit (s) per request batch
    static final double PRICE_PER_VM_HOUR = 0.05;   // USD, assumed price

    static final double UP_THRESHOLD = 0.80;        // reactive: scale up above this utilisation
    static final double DOWN_THRESHOLD = 0.40;      // reactive: scale down below this
    static final double REACTIVE_TARGET = 0.70;     // reactive: utilisation to scale towards
    static double SAFETY_MARGIN = 0.40;             // predictive: buffer on top of the forecast (arg 2)

    // MI of one request so that VM_CAPACITY requests exactly fill one VM for one step
    static final long MI_PER_REQUEST = (long) (VM_MIPS * STEP / VM_CAPACITY);

    enum Policy { STATIC, REACTIVE, PREDICTIVE_MA, PREDICTIVE_LSTM }

    // workload columns from predictions.csv
    static int[] actual, lstmPred, maPred;
    static String[] timestamps;
    static int maxVms;

    public static void main(String[] args) throws IOException {
        Log.setLevel(Level.WARN);   // keep CloudSim's own logs quiet
        loadCsv(args.length > 0 ? args[0] : CSV_PATH);
        if (args.length > 1) SAFETY_MARGIN = Double.parseDouble(args[1]);
        maxVms = (int) Math.ceil((double) Arrays.stream(actual).max().orElse(0) / VM_CAPACITY) + 2;
        Files.createDirectories(Path.of(OUT_DIR));

        System.out.printf("Loaded %d steps (%s -> %s), %d total requests%n",
                actual.length, timestamps[0], timestamps[actual.length - 1],
                Arrays.stream(actual).sum());
        System.out.printf("VM capacity = %d req/step, VM pool = %d, boot time = %.0f s, safety margin = %.0f%%%n%n",
                VM_CAPACITY, maxVms, BOOT_TIME, SAFETY_MARGIN * 100);

        try (PrintWriter summary = new PrintWriter(OUT_DIR + "/summary.csv")) {
            summary.println("policy,vm_hours,avg_vms,cost_usd,avg_response_s,p95_response_s,"
                    + "sla_violation_pct,underprovisioned_steps_pct,avg_utilization_pct");
            System.out.printf("%-16s %9s %8s %9s %10s %10s %9s %10s %8s%n",
                    "Policy", "VM-hours", "AvgVMs", "Cost($)", "AvgResp(s)", "P95Resp(s)",
                    "SLAviol%", "UnderProv%", "Util%");
            for (Policy p : Policy.values()) {
                double[] r = runPolicy(p);
                summary.printf("%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f%n",
                        p, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
                System.out.printf("%-16s %9.1f %8.2f %9.2f %10.2f %10.2f %9.2f %10.2f %8.1f%n",
                        p, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
            }
        }
        System.out.println("\nResults written to ./" + OUT_DIR + "/");
    }

    /** Decide how many VMs should be active in each step for a policy. */
    static int[] scalingDecisions(Policy p) {
        int n = actual.length;
        int[] target = new int[n];
        int current = Math.max(1, (int) Math.ceil(actual[0] / (VM_CAPACITY * REACTIVE_TARGET)));
        for (int t = 0; t < n; t++) {
            switch (p) {
                case STATIC -> current = (int) Math.ceil(
                        (double) Arrays.stream(actual).max().orElse(0) / VM_CAPACITY);
                case REACTIVE -> {
                    if (t > 0) {   // only knows the load that already happened
                        double util = actual[t - 1] / (current * (double) VM_CAPACITY);
                        if (util > UP_THRESHOLD || util < DOWN_THRESHOLD)
                            current = (int) Math.ceil(actual[t - 1] / (VM_CAPACITY * REACTIVE_TARGET));
                    }
                }
                case PREDICTIVE_MA -> current = (int) Math.ceil(maPred[t] * (1 + SAFETY_MARGIN) / VM_CAPACITY);
                case PREDICTIVE_LSTM -> current = (int) Math.ceil(lstmPred[t] * (1 + SAFETY_MARGIN) / VM_CAPACITY);
            }
            current = Math.max(1, Math.min(current, maxVms));
            target[t] = current;
        }
        return target;
    }

    /** Runs one CloudSim Plus simulation. Returns the metrics row. */
    static double[] runPolicy(Policy policy) throws IOException {
        int n = actual.length;
        int[] target = scalingDecisions(policy);

        CloudSimPlus simulation = new CloudSimPlus();
        createDatacenter(simulation);
        DatacenterBroker broker = new DatacenterBrokerSimple(simulation);
        broker.setVmDestructionDelay(-1);   // keep pool VMs alive while idle

        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < maxVms; i++) {
            Vm vm = new VmSimple(VM_MIPS, 1);
            vm.setCloudletScheduler(new CloudletSchedulerSpaceShared());   // FIFO queue per VM
            vm.setRam(1024).setBw(1000).setSize(10_000);
            vms.add(vm);
        }
        broker.submitVmList(vms);

        // readyAt[i] = time VM i can start serving (after boot); active = index < target[t]
        double[] readyAt = new double[maxVms];
        for (int i = 0; i < target[0]; i++) readyAt[i] = 0;
        int prev = target[0];

        List<Cloudlet> cloudlets = new ArrayList<>();
        double activeVmSeconds = 0, servableCapacity = 0;
        int underProvisioned = 0, rr = 0;

        try (PrintWriter tl = new PrintWriter(OUT_DIR + "/timeline_" + policy + ".csv")) {
            tl.println("step,timestamp,actual_requests,forecast,active_vms,ready_capacity");
            for (int t = 0; t < n; t++) {
                double stepStart = t * STEP;
                if (t > 0 && target[t] > prev)              // scale-up: new VMs start booting now
                    for (int i = prev; i < target[t]; i++) readyAt[i] = stepStart + BOOT_TIME;
                prev = target[t];
                activeVmSeconds += target[t] * STEP;

                double capacity = 0;   // requests the ready VMs can serve this step
                for (int i = 0; i < target[t]; i++)
                    capacity += VM_CAPACITY * (STEP - Math.min(STEP, Math.max(0, readyAt[i] - stepStart))) / STEP;
                servableCapacity += capacity;
                if (actual[t] > capacity) underProvisioned++;

                int forecast = switch (policy) {
                    case PREDICTIVE_LSTM -> lstmPred[t];
                    case PREDICTIVE_MA -> maPred[t];
                    default -> -1;
                };
                tl.printf("%d,%s,%d,%d,%d,%.1f%n", t, timestamps[t], actual[t], forecast, target[t], capacity);

                // create request batches arriving evenly over this step
                int batches = (int) Math.ceil(actual[t] / (double) BATCH);
                int remaining = actual[t];
                for (int b = 0; b < batches; b++) {
                    int reqs = Math.min(BATCH, remaining);
                    remaining -= reqs;
                    double arrival = stepStart + b * STEP / batches;

                    // load balancer: round-robin over active VMs that have finished booting
                    Vm chosen = null;
                    for (int k = 0; k < target[t]; k++) {
                        int idx = (rr + k) % target[t];
                        if (readyAt[idx] <= arrival) { chosen = vms.get(idx); rr = idx + 1; break; }
                    }
                    if (chosen == null) chosen = vms.get(0);

                    Cloudlet c = new CloudletSimple(MI_PER_REQUEST * reqs, 1);
                    c.setUtilizationModelCpu(new UtilizationModelFull())
                     .setUtilizationModelRam(new UtilizationModelDynamic(0))
                     .setUtilizationModelBw(new UtilizationModelDynamic(0))
                     .setSizes(300);
                    c.setSubmissionDelay(arrival);
                    c.setVm(chosen);
                    cloudlets.add(c);
                }
            }
        }

        broker.submitCloudletList(cloudlets);
        simulation.start();

        List<Cloudlet> finished = broker.getCloudletFinishedList();
        double[] resp = finished.stream()
                .mapToDouble(c -> c.getFinishTime() - c.getSubmissionDelay())
                .sorted().toArray();
        long violations = Arrays.stream(resp).filter(r -> r > SLA_RESPONSE_TIME).count();
        if (finished.size() < cloudlets.size())
            System.out.printf("  [warn] %s: %d of %d cloudlets did not finish%n",
                    policy, cloudlets.size() - finished.size(), cloudlets.size());

        double vmHours = activeVmSeconds / 3600.0;
        double totalReq = Arrays.stream(actual).sum();
        return new double[]{
                vmHours,
                activeVmSeconds / (n * STEP),
                vmHours * PRICE_PER_VM_HOUR,
                Arrays.stream(resp).average().orElse(0),
                resp.length == 0 ? 0 : resp[(int) Math.min(resp.length - 1, Math.ceil(0.95 * resp.length) - 1)],
                100.0 * violations / Math.max(1, resp.length),
                100.0 * underProvisioned / n,
                100.0 * totalReq / Math.max(1, servableCapacity)
        };
    }

    /** One datacenter with a few multi-core hosts, enough for the whole VM pool. */
    static void createDatacenter(CloudSimPlus sim) {
        List<Host> hosts = new ArrayList<>();
        int hostsNeeded = (int) Math.ceil(maxVms / 8.0);
        for (int h = 0; h < hostsNeeded; h++) {
            List<Pe> pes = new ArrayList<>();
            for (int i = 0; i < 8; i++) pes.add(new PeSimple(VM_MIPS));
            hosts.add(new HostSimple(16_384, 100_000, 1_000_000, pes));
        }
        new DatacenterSimple(sim, hosts);
    }

    static void loadCsv(String path) throws IOException {
        List<String> lines = Files.readAllLines(Path.of(path));
        List<String> header = Arrays.asList(lines.get(0).trim().split(","));
        int iTs = header.indexOf("timestamp"), iA = header.indexOf("actual_requests"),
            iL = header.indexOf("lstm_pred"), iM = header.indexOf("ma_pred");
        int n = lines.size() - 1;
        actual = new int[n]; lstmPred = new int[n]; maPred = new int[n]; timestamps = new String[n];
        for (int r = 0; r < n; r++) {
            String[] f = lines.get(r + 1).trim().split(",");
            timestamps[r] = f[iTs];
            actual[r] = Integer.parseInt(f[iA]);
            lstmPred[r] = Math.max(0, Integer.parseInt(f[iL]));
            maPred[r] = Math.max(0, Integer.parseInt(f[iM]));
        }
    }
}
