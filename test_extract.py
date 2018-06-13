import matlab.engine as meng

eng = meng.start_matlab()

f = eng.extract_features('music/147526804.wav')
print(f)

eng.quit()
