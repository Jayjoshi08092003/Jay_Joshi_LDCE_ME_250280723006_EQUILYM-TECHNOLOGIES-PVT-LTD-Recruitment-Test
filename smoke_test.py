from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
print("default run OK; metrics:", [m.value for m in at.metric])
# live filter: only Mehsana
at.sidebar.multiselect[0].set_value(["Mehsana"]).run()
assert not at.exception, at.exception
print("Mehsana only:", [m.value for m in at.metric])
# raise trend threshold to 40 %
at.sidebar.multiselect[0].set_value(sorted(["Ahmedabad","Surat","Vadodara","Rajkot","Mehsana","Bhavnagar"])).run()
at.sidebar.slider[0].set_value(40.0).run()
assert not at.exception, at.exception
print("trend>=40%:", [m.value for m in at.metric])
# z-score
at.sidebar.radio[0].set_value("zscore").run()
assert not at.exception, at.exception
print("zscore:", [m.value for m in at.metric])
