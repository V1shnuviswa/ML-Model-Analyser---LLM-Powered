import React from "react";

const buttonStyle = {
  position: "fixed",
  bottom: "20px",
  right: "20px",
  padding: "12px 24px",
  fontSize: "16px",
  fontWeight: "bold",
  background: "linear-gradient(135deg, #4f46e5, #3b82f6)",
  color: "white",
  border: "none",
  borderRadius: "10px",
  cursor: "pointer",
  boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.2)",
  transition: "background 0.3s ease, transform 0.2s ease",
  zIndex: 1000,
};

const hoverStyle = {
  background: "linear-gradient(135deg, #4338ca, #2563eb)",
  transform: "scale(1.05)",
};

const activeStyle = {
  background: "#372ec2",
  transform: "scale(0.98)",
};

class DownloadButton extends React.Component {
  state = {
    isHovered: false,
    isActive: false,
  };

  handleDownload = () => {
    fetch("http://localhost:5000/generate_pdf")
      .then((response) => response.blob())
      .then((blob) => {
        const link = document.createElement("a");
        link.href = window.URL.createObjectURL(blob);
        link.download = "ML_Project_Analysis.pdf";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      })
      .catch((error) => console.error("Error downloading PDF:", error));
  };

  render() {
    const { isHovered, isActive } = this.state;

    const combinedStyle = {
      ...buttonStyle,
      ...(isHovered ? hoverStyle : {}),
      ...(isActive ? activeStyle : {}),
    };

    return (
      <button
        style={combinedStyle}
        onMouseEnter={() => this.setState({ isHovered: true })}
        onMouseLeave={() => this.setState({ isHovered: false, isActive: false })}
        onMouseDown={() => this.setState({ isActive: true })}
        onMouseUp={() => this.setState({ isActive: false })}
        onClick={this.handleDownload}
      >
        Download Report PDF
      </button>
    );
  }
}

export default DownloadButton;
