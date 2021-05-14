import os, pandas as pd, shutil
from pathlib import Path
from typing import Optional, Dict, Union

from grams.misc import get_incremental_path, measure_time, deserialize_csv, serialize_csv, serialize_json
from grams.misc.snapshot import snapshot


class ExpManager:
    def __init__(self, workdir: Union[Path, str], exp_name: str, exp_params: dict, integrate_neptune: bool = False):
        # directory containing the experiments
        self.workdir = Path(workdir)
        # name of the experiment: e.g., baseline, proposed-method-a, proposed-method-b
        self.exp_name = exp_name
        # parameters of the experiment
        self.exp_params = exp_params
        self.current_run_dir = None

        # whether we should integrating with neptune AI to track experiment run.
        self.integrate_neptune = integrate_neptune
        if integrate_neptune:
            import neptune
            self.neptune = neptune
            def neptune_fn():
                self.neptune.init(project_qualified_name=os.environ['NEPTUNE_PROJECT'], api_token=os.environ['NEPTUNE_API_TOKEN'])
            print("Setup neptune take: %.3f seconds" % measure_time(neptune_fn))

        assert self.workdir.exists(), f"{self.workdir} does not exist"

    def create_run(self, run_id: Optional[str] = None):
        """Create new directory to store the run

        Parameters
        ----------
        run_id: optional run id (for reuse previous run)

        Returns
        -------
        the run directory
        """
        if run_id is not None:
            self.current_run_dir = self.workdir / run_id
        else:
            self.current_run_dir = Path(get_incremental_path(self.workdir / "v"))
        self.current_run_dir.mkdir(exist_ok=True, parents=True)
        serialize_json(self.exp_params, self.current_run_dir / "exp_params.json", indent=4)

        if self.integrate_neptune:
            def neptune_fn():
                neptune_exp_params = dict()
                neptune_exp_params.update(self.exp_params)
                neptune_exp_params['workdir'] = self.workdir
                neptune_exp_params['run_id'] = self.current_run_dir.name
                self.neptune.create_experiment(
                    name=self.exp_name, params=neptune_exp_params, 
                    send_hardware_metrics=False, run_monitoring_thread=False)
            print("Create neptune experiment take: %.3f seconds" % measure_time(neptune_fn))
        return self.current_run_dir

    def snapshot(self):
        assert self.current_run_dir is not None, "Need to create a run first."
        snapshot(self.current_run_dir)

    def report_metrics(self, metrics: Dict[str, Union[str, int, float]], filename: str = "metrics.csv"):
        if self.integrate_neptune:
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    self.neptune.log_metric(k, v)
                else:
                    self.neptune.log_text(k, v)

        # log the data to a file
        pd.DataFrame([
            {"name": k, "value": v}
            for k, v in metrics.items()
        ]).to_csv(self.current_run_dir / filename, columns=["name", "value"])


class ExpResults:
    """Storing the experiment result in tsv/csv file.
    Old result is override by new result.
    Always keep a backup.
    """
    def __init__(self, outfile: Union[Path, str]):
        self.outfile = Path(outfile)
        self.records = {}
        self.headers = None
        assert self.outfile.name.endswith(".csv") or self.outfile.name.endswith(".tsv")
        if self.outfile.name.endswith(".csv"):
            self.delimiter = ","
        else:
            self.delimiter = "\t"

        if self.outfile.exists():
            records = deserialize_csv(self.outfile, self.delimiter)
            self.headers = records[0]
            for record in records[1:]:
                record = dict(zip(self.headers, record))
                self.records[record['id']] = record
            shutil.copyfile(str(self.outfile), str(self.outfile) + ".backup")
        
    def add(self, id: str, result: dict):
        """Add an result, identified by id, the result dict should not have id"""
        # copy record to keep the header in order, including new field called id
        new_record = {'id': id}
        for k, v in result.items():
            new_record[k] = v
        if self.headers is None:
            self.headers = list(new_record.keys())
            serialize_csv([self.headers], self.outfile, mode="a", delimiter=self.delimiter)    

        self.records[id] = new_record
        serialize_csv([[new_record[k] for k in self.headers]], self.outfile, mode="a", delimiter=self.delimiter)
    
    def get_dataframe(self):
        return pd.DataFrame(self.records)


if __name__ == '__main__':
    import neptune
    neptune.init(project_qualified_name=os.environ['NEPTUNE_PROJECT'], api_token=os.environ['NEPTUNE_API_TOKEN'])
    neptune_exp_params = dict(test=True, name='PSL')
    neptune.create_experiment(name='Tester', params=neptune_exp_params)
    neptune.log_metric("noigi", 500)