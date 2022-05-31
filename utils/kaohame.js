const run = async () => {
  new Promise(() => {
    let faceImage = document.getElementById("faceSrc");
    let back = cv.imread("noFace");
    let face = cv.imread(faceImage);
    let face_h = face.rows;
    let face_w = face.cols;

    let back_h = back.rows;
    let back_w = back.cols;

    let copy = new cv.Mat();

    let low = new cv.Mat(back_h, back_w, back.type(), [254, 254, 254, 0]);
    let high = new cv.Mat(back_h, back_w, back.type(), [255, 255, 255, 255]);
    let back_white = new cv.Mat();
    cv.inRange(back, low, high, back_white);

    //edge detection for coastline========================================
    let edge = new cv.Mat();
    cv.Canny(face, edge, 30, 80, 3, false); //cvtColor(mat,dst,cv.COLOR_RGBA2GRAY);

    copy = edge.clone();

    let back_ori = [0, 0];
    while (back_white.ucharPtr(back_ori[0], back_ori[1])[0] === 0) {
      back_ori[1]++;
      if (back_ori[1] === back_w) {
        back_ori[0]++;
        back_ori[1] = 0;
      }
    }
    //back_white.ucharPtr(back_ori[0],back_ori[1])[0]=0;

    let back_l = back_ori[1];
    let back_r = back_ori[1];
    let back_t = back_ori[0];
    let back_b = back_ori[0];

    let back_len = 0;
    let back_newlen = 0;

    while (back_newlen >= back_len) {
      back_len = back_newlen;
      back_l = back_ori[1];
      while (back_white.ucharPtr(back_b, back_l)[0] === 255) back_l--;

      while (back_white.ucharPtr(back_b, back_r)[0] === 255) back_r++;

      back_newlen = back_r - back_l;
      back_b++;
    }

    back_ori[1] = Math.floor((back_l + back_r) / 2);

    while (back_white.ucharPtr(back_b, back_ori[1])[0] === 255) back_b++;

    for (let i = 0; i < face_h; i++) {
      for (let j = 0; j < face_w; j++) {
        if (edge.ucharPtr(i, j)[0] === 255) {
          for (let k = 0; k < 3; k++) {
            copy.ucharPtr(i + k, j)[0] = 255;
            copy.ucharPtr(i - k, j)[0] = 255;
            copy.ucharPtr(i + k, j + k)[0] = 255;
            copy.ucharPtr(i - k, j - k)[0] = 255;
            copy.ucharPtr(i + k, j - k)[0] = 255;
            copy.ucharPtr(i - k, j + k)[0] = 255;
          }
        }
      }
    }

    //===============================================
    let mark = cv.Mat.zeros(face_h, face_w, edge.type());
    let arr = [[0, 0]];
    let pos = [0, 0];
    mark.ucharPtr(pos[0], pos[1])[0] = 255;

    while (arr.length !== 0) {
      pos = arr.shift();
      if (copy.ucharPtr(pos[0], pos[1])[0] === 0) {
        if (mark.ucharPtr(pos[0] + 1, pos[1])[0] === 0) {
          mark.ucharPtr(pos[0] + 1, pos[1])[0] = 255;
          arr.push([pos[0] + 1, pos[1]]);
        }
        if (mark.ucharPtr(pos[0], pos[1] + 1)[0] === 0) {
          mark.ucharPtr(pos[0], pos[1] + 1)[0] = 255;
          arr.push([pos[0], pos[1] + 1]);
        }

        if (pos[0] > 0 && mark.ucharPtr(pos[0] - 1, pos[1])[0] === 0) {
          mark.ucharPtr(pos[0] - 1, pos[1])[0] = 255;
          arr.push([pos[0] - 1, pos[1]]);
        }
        if (pos[1] > 0 && mark.ucharPtr(pos[0], pos[1] - 1)[0] === 0) {
          mark.ucharPtr(pos[0], pos[1] - 1)[0] = 255;
          arr.push([pos[0], pos[1] - 1]);
        }
      }
    }

    let main = cv.Mat.zeros(face_h, face_w, edge.type());
    for (let i = 0; i < face_h; i++) {
      for (let j = 0; j < face_w; j++) {
        if (mark.ucharPtr(i, j)[0] === 0) {
          main.ucharPtr(i, j)[0] = 255;
        }
      }
    }

    copy = face.clone();
    for (let i = 0; i < face_h; i++) {
      for (let j = 0; j < face_w; j++) {
        if (main.ucharPtr(i, j)[0] === 0) {
          copy.ucharPtr(i, j)[0] = 231;
          copy.ucharPtr(i, j)[1] = 209;
          copy.ucharPtr(i, j)[2] = 183;
        }
      }
    }

    let coast = new cv.Mat();
    cv.Canny(mark, coast, 30, 80, 3, false);

    let midh = Math.floor(face_h / 2);
    let midw = Math.floor(face_w / 2);
    let l = midw;
    let r = midw;
    let t = midh;
    let len = 0;
    let newlen = 1;

    while (newlen >= len) {
      len = newlen;
      l = midw;
      r = midw;
      while (coast.ucharPtr(t, l)[0] === 0) {
        l--;
      }

      while (coast.ucharPtr(t, r)[0] === 0) {
        coast.ucharPtr(t, r)[0] = 255;
        r++;
      }
      newlen = r - l;
      t--;
    }

    midw = Math.floor((r + l) / 2);
    midh = t;

    while (coast.ucharPtr(t, midw)[0] === 0) t--;

    let b = 2 * midh - t;

    let pure_face = new cv.Mat();

    let rect = new cv.Rect(l, t, r - l, b - t);

    pure_face = face.roi(rect);

    let size = new cv.Size(back_r - back_l, back_b - back_t);

    cv.resize(pure_face, pure_face, size, 0, 0, cv.INTER_AREA);

    let back_copy = back.clone();

    for (let i = 0; i < back_b - back_t; i++) {
      for (let j = 0; j < back_r - back_l; j++) {
        if (back_white.ucharPtr(back_t + i, back_l + j)[0] === 255) {
          back_copy.ucharPtr(back_t + i, back_l + j)[0] = pure_face.ucharPtr(
            i,
            j
          )[0];
          back_copy.ucharPtr(back_t + i, back_l + j)[1] = pure_face.ucharPtr(
            i,
            j
          )[1];
          back_copy.ucharPtr(back_t + i, back_l + j)[2] = pure_face.ucharPtr(
            i,
            j
          )[2];
        }
      }
    }

    cv.imshow("back", back_copy);
    // cv.imshow("face", pure_face);

    face.delete();
    back.delete();
    edge.delete();
    copy.delete();
    mark.delete();
    coast.delete();
    pure_face.delete();
    back_copy.delete();
  });
};
