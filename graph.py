import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------- Compliance Constants Calculation -------- #
def calculate_compliance(C11, C12, C44):
    S11 = (C11 + C12) / ((C11 - C12) * (C11 + 2*C12))
    S12 = -C12 / ((C11 - C12) * (C11 + 2*C12))
    S44 = 1 / C44
    return S11, S12, S44

# -------- Plotting Functions -------- #
def plot_3d_property(property_name, S11, S12, S44):
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    l = np.sin(theta) * np.cos(phi)
    m = np.sin(theta) * np.sin(phi)
    n = np.cos(theta)

    if property_name == "Young's modulus":
        H = (l**2 * m**2 + m**2 * n**2 + n**2 * l**2)
        inv_E = S11 - 2 * (S11 - S12 - 0.5 * S44) * H
        prop = 1 / inv_E
        label = "Young's Modulus (GPa)"
    elif property_name == "Shear modulus":
        H = (l**2 * m**2 + m**2 * n**2 + n**2 * l**2)
        inv_G = S44 + 0.5 * (S11 - S12 - 0.5 * S44) * H
        prop = 1 / inv_G
        label = "Shear Modulus (GPa)"
    elif property_name == "Poisson's ratio":
        numerator = -(S11 + S12 - 2 * S44) * (l**2 * m**2 + m**2 * n**2 + n**2 * l**2)
        denominator = (S11 - S12)
        prop = numerator / denominator
        label = "Poisson's Ratio"
    else:
        return

    x, y, z = prop * l, prop * m, prop * n

    fig = plt.figure(figsize=(9, 7))  # Larger figure size
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(
        x, y, z,
        facecolors=plt.cm.jet(prop / np.nanmax(prop)),
        rstride=1, cstride=1, linewidth=0.3, edgecolor='k', alpha=0.95,
    )

    mappable = plt.cm.ScalarMappable(cmap='jet')
    mappable.set_array(prop)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.10, shrink=0.7, aspect=15)  # Increased pad
    cbar.set_label(label, fontsize=14)

    ax.set_xlabel('X', labelpad=8, fontsize=15)  # Increased X label padding
    ax.set_ylabel('Y', labelpad=8 , fontsize=15)  # Increased Y label padding
    ax.set_zlabel('Z', labelpad=8, fontsize=15)  # Increased Z label padding

   # Manually set tick label sizes
    ax.set_xticklabels([f"{tick:.1f}" for tick in ax.get_xticks()], fontsize=12)
    ax.set_yticklabels([f"{tick:.1f}" for tick in ax.get_yticks()], fontsize=12)
    ax.set_zticklabels([f"{tick:.1f}" for tick in ax.get_zticks()], fontsize=12)

    st.pyplot(fig)

def plot_2d_property(property_name, S11, S12, S44):
    theta = np.linspace(0, 2*np.pi, 360)
    l = np.cos(theta)
    m = np.sin(theta)
    n = 0

    if property_name == "Young's modulus":
        H = (l**2 * m**2)
        inv_E = S11 - 2 * (S11 - S12 - 0.5 * S44) * H
        prop = 1 / inv_E
        label = "Young's Modulus (GPa)"
    elif property_name == "Shear modulus":
        H = (l**2 * m**2)
        inv_G = S44 + 0.5 * (S11 - S12 - 0.5 * S44) * H
        prop = 1 / inv_G
        label = "Shear Modulus (GPa)"
    elif property_name == "Poisson's ratio":
        numerator = -(S11 + S12 - 2*S44) * (l**2 * m**2)
        denominator = (S11 - S12)
        prop = numerator / denominator
        label = "Poisson's Ratio"
    else:
        return

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(theta, prop, color='b', linewidth=2)
    ax.fill(theta, prop, color='skyblue', alpha=0.4)
    ax.set_title(label, fontsize=16)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    # -------- Increase font size of radial tick labels -------- #
    for label in ax.get_yticklabels():
      label.set_fontsize(14)  # Adjust to your desired font size

# -------- Increase font size of angular (theta) tick labels -------- #
    for label in ax.get_xticklabels():
      label.set_fontsize(14)

    st.pyplot(fig)

# -------- Streamlit GUI -------- #
st.title("Elastic Property Visualizer")

C11 = st.number_input("Insert value of C11:", format="%.5f", step=0.1)
C12 = st.number_input("Insert value of C12:", format="%.5f", step=0.1)
C44 = st.number_input("Insert value of C44:", format="%.5f", step=0.1)

property_name = st.selectbox("Select Property:", ["Young's modulus", "Shear modulus", "Poisson's ratio"])
dimension = st.selectbox("Select Dimension:", ["2D", "3D"])

if st.button("Generate Plot"):
    if C11 > 0 and C44 > 0 and C12 <= C11:
        S11, S12, S44 = calculate_compliance(C11, C12, C44)
        if dimension == "2D":
            plot_2d_property(property_name, S11, S12, S44)
        else:
            plot_3d_property(property_name, S11, S12, S44)
    else:
        st.error("Please enter valid values for C11, C12, and C44.")
        st.warning("Ensure C11 and C44 are positive and C12 <= C11.")